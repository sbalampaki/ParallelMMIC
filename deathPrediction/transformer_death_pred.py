#!/usr/bin/env python3
"""
Transformer-based Clinical Prediction for Patient Mortality
===========================================================
Implements a tabular transformer that learns rich feature interactions
through multi-head self-attention for hospital mortality prediction
using the MIMIC-derived dataset.

Usage:
    python transformer_death_pred.py mimic_data.csv [--epochs 50] [--batch-size 64]
                                                     [--lr 1e-3] [--heads 4]
                                                     [--layers 2] [--embed-dim 64]
                                                     [--dropout 0.1] [--seed 42]
"""

import argparse
import csv
import math
import random
import sys
import time
from collections import defaultdict

# ── Optional torch import (fail gracefully) ───────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str):
    """
    Load MIMIC CSV and return a list of dicts with selected fields.

    CSV columns expected:
        SUBJECT_ID, HADM_ID, HOSPITAL_EXPIRE_FLAG, ADMITTIME,
        ETHNICITY, GENDER, DOB, AGE_AT_ADMISSION, ICD9_CODE_1, ...
    """
    records = []
    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                expire_flag = int(row.get("HOSPITAL_EXPIRE_FLAG", "0").strip() or 0)
                age = float(row.get("AGE_AT_ADMISSION", "0").strip() or 0)
                icd9 = int(row.get("ICD9_CODE_1", "0").strip() or 0)
                ethnicity = row.get("ETHNICITY", "OTHER").strip()
                gender = row.get("GENDER", "M").strip()
                records.append(
                    {
                        "expire_flag": expire_flag,
                        "age": age,
                        "icd9_code1": icd9,
                        "ethnicity": ethnicity,
                        "gender": gender,
                    }
                )
            except (ValueError, KeyError):
                continue
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Feature engineering & vocabulary building
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(records):
    """
    Build integer vocabularies for categorical features and compute
    normalization statistics for continuous features.
    """
    eth_vocab = {"<UNK>": 0}
    gen_vocab = {"<UNK>": 0}
    icd_vocab = {"<UNK>": 0}

    for r in records:
        if r["ethnicity"] not in eth_vocab:
            eth_vocab[r["ethnicity"]] = len(eth_vocab)
        if r["gender"] not in gen_vocab:
            gen_vocab[r["gender"]] = len(gen_vocab)
        icd_key = str(r["icd9_code1"])
        if icd_key not in icd_vocab:
            icd_vocab[icd_key] = len(icd_vocab)

    # Continuous feature statistics (age normalisation)
    ages = [r["age"] for r in records]
    age_mean = sum(ages) / max(len(ages), 1)
    age_std = math.sqrt(
        sum((a - age_mean) ** 2 for a in ages) / max(len(ages), 1)
    ) or 1.0

    return eth_vocab, gen_vocab, icd_vocab, age_mean, age_std


def encode_record(record, eth_vocab, gen_vocab, icd_vocab, age_mean, age_std):
    """Return (cat_ids, cont_feats, label) for a single patient record."""
    eth_id = eth_vocab.get(record["ethnicity"], 0)
    gen_id = gen_vocab.get(record["gender"], 0)
    icd_id = icd_vocab.get(str(record["icd9_code1"]), 0)
    age_norm = (record["age"] - age_mean) / age_std
    label = record["expire_flag"]
    return [eth_id, gen_id, icd_id], [age_norm], label


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalDataset(Dataset):
    """Wraps encoded patient records as a PyTorch Dataset."""

    def __init__(self, records, eth_vocab, gen_vocab, icd_vocab, age_mean, age_std):
        self.samples = []
        for r in records:
            cat_ids, cont_feats, label = encode_record(
                r, eth_vocab, gen_vocab, icd_vocab, age_mean, age_std
            )
            self.samples.append((cat_ids, cont_feats, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat_ids, cont_feats, label = self.samples[idx]
        return (
            torch.tensor(cat_ids, dtype=torch.long),
            torch.tensor(cont_feats, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Transformer model
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalTransformer(nn.Module):
    """
    Tabular transformer for binary clinical outcome prediction.

    Architecture:
      ┌─────────────────────────────────────────────────────────────┐
      │  Categorical features → Embedding layers                     │
      │  Continuous features  → Linear projection                    │
      │  [CLS] token          → Learnable classification token       │
      │                                                               │
      │  Concatenate all feature tokens → (B, num_tokens, embed_dim)│
      │                                                               │
      │  Transformer encoder blocks (N × multi-head self-attention + │
      │                              feed-forward + layer-norm)      │
      │                                                               │
      │  Extract [CLS] token output → Linear → sigmoid → P(death)   │
      └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        eth_vocab_size: int,
        gen_vocab_size: int,
        icd_vocab_size: int,
        num_cont_features: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Categorical embeddings (one embedding table per feature)
        self.eth_embed = nn.Embedding(eth_vocab_size, embed_dim, padding_idx=0)
        self.gen_embed = nn.Embedding(gen_vocab_size, embed_dim, padding_idx=0)
        self.icd_embed = nn.Embedding(icd_vocab_size, embed_dim, padding_idx=0)

        # Continuous feature projection: map each scalar to embed_dim
        self.cont_proj = nn.Linear(num_cont_features, embed_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional encodings for num_tokens = 1 (CLS) + 3 cat + 1 cont
        num_tokens = 1 + 3 + 1
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, cat_ids, cont_feats):
        """
        Parameters
        ----------
        cat_ids   : (B, 3)  – [ethnicity_id, gender_id, icd9_id]
        cont_feats: (B, C)  – continuous features (e.g. normalised age)
        """
        B = cat_ids.size(0)

        # Embed categorical features → (B, 1, embed_dim) each
        eth_tok = self.eth_embed(cat_ids[:, 0]).unsqueeze(1)
        gen_tok = self.gen_embed(cat_ids[:, 1]).unsqueeze(1)
        icd_tok = self.icd_embed(cat_ids[:, 2]).unsqueeze(1)

        # Project continuous features → (B, 1, embed_dim)
        cont_tok = self.cont_proj(cont_feats).unsqueeze(1)

        # Expand [CLS] token to batch
        cls_tok = self.cls_token.expand(B, -1, -1)

        # Concatenate: [CLS, eth, gen, icd, cont] → (B, 5, embed_dim)
        x = torch.cat([cls_tok, eth_tok, gen_tok, icd_tok, cont_tok], dim=1)
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.transformer(x)

        # Use [CLS] token for classification
        cls_out = x[:, 0, :]

        # Binary classification logit → probability
        logit = self.classifier(cls_out).squeeze(-1)
        return logit  # Return logit; apply sigmoid outside for BCEWithLogitsLoss


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training & evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for cat_ids, cont_feats, labels in loader:
        cat_ids    = cat_ids.to(device)
        cont_feats = cont_feats.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()
        logits = model(cat_ids, cont_feats)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * cat_ids.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_pred_death = 0.0
    total = 0
    tp = fp = tn = fn = 0

    for cat_ids, cont_feats, labels in loader:
        cat_ids    = cat_ids.to(device)
        cont_feats = cont_feats.to(device)
        labels_np  = labels.numpy()

        logits = model(cat_ids, cont_feats).cpu()
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs >= 0.5).astype(int)

        for i in range(len(labels_np)):
            y = int(labels_np[i])
            p = int(preds[i])
            if p == y:
                correct += 1
            if p == 1 and y == 1:
                tp += 1
            elif p == 1 and y == 0:
                fp += 1
            elif p == 0 and y == 1:
                fn += 1
            else:
                tn += 1
            total_pred_death += float(probs[i])
            total += 1

    accuracy  = correct / max(total, 1)
    death_rate = total_pred_death / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return accuracy, death_rate, precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Transformer-based clinical mortality prediction"
    )
    parser.add_argument("data_file", help="Path to MIMIC CSV file")
    parser.add_argument("--epochs",     type=int,   default=50,   help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Mini-batch size (default: 64)")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--heads",      type=int,   default=4,    help="Attention heads (default: 4)")
    parser.add_argument("--layers",     type=int,   default=2,    help="Transformer layers (default: 2)")
    parser.add_argument("--embed-dim",  type=int,   default=64,   help="Embedding dimension (default: 64)")
    parser.add_argument("--dropout",    type=float, default=0.1,  help="Dropout rate (default: 0.1)")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed (default: 42)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is not installed.")
        print("Install with:  pip install torch")
        sys.exit(1)

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Transformer-based Clinical Prediction")
    print("======================================")
    print(f"Device: {device}")

    # ── Load & encode data ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    records = load_data(args.data_file)
    load_time = time.perf_counter() - t0

    if not records:
        print("Error: No records loaded – check the CSV path and format.")
        sys.exit(1)

    print(f"Data loaded: {len(records)} patients  ({load_time:.4f}s)")

    eth_vocab, gen_vocab, icd_vocab, age_mean, age_std = build_vocab(records)
    print(
        f"Vocabulary sizes – ethnicity: {len(eth_vocab)}, "
        f"gender: {len(gen_vocab)}, ICD9: {len(icd_vocab)}"
    )

    # ── Dataset split (80 / 20) ───────────────────────────────────────────────
    dataset = ClinicalDataset(records, eth_vocab, gen_vocab, icd_vocab, age_mean, age_std)
    train_n = int(0.8 * len(dataset))
    test_n  = len(dataset) - train_n
    train_ds, test_ds = random_split(
        dataset,
        [train_n, test_n],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    print(f"Training samples: {train_n}  Test samples: {test_n}")
    print(
        f"Model: Transformer (layers={args.layers}, heads={args.heads}, "
        f"embed_dim={args.embed_dim}, dropout={args.dropout})"
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = ClinicalTransformer(
        eth_vocab_size=len(eth_vocab),
        gen_vocab_size=len(gen_vocab),
        icd_vocab_size=len(icd_vocab),
        num_cont_features=1,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        ffn_dim=args.embed_dim * 2,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    # Use pos_weight to handle class imbalance (rare deaths)
    pos_count    = sum(r["expire_flag"] for r in records)
    neg_count    = len(records) - pos_count
    pos_weight   = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    t_train_start = time.perf_counter()
    print("\nEpoch  Loss       LR")
    print("─" * 30)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  {epoch:4d}  {loss:.6f}  {lr_now:.2e}")

    train_time = time.perf_counter() - t_train_start
    print(f"\nTraining time: {train_time:.4f} seconds")

    # ── Evaluation ────────────────────────────────────────────────────────────
    t_eval_start = time.perf_counter()
    accuracy, death_rate, precision, recall, f1 = evaluate(model, test_loader, device)
    eval_time = time.perf_counter() - t_eval_start

    total_time = load_time + train_time + eval_time

    print(f"Evaluation time: {eval_time:.4f} seconds")
    print(f"Total execution time: {total_time:.4f} seconds")
    print("\nResults:")
    print(f"  Accuracy:             {accuracy * 100:.2f}%")
    print(f"  Predicted Death Rate: {death_rate * 100:.2f}%")
    print(f"  Precision:            {precision * 100:.2f}%")
    print(f"  Recall:               {recall * 100:.2f}%")
    print(f"  F1 Score:             {f1 * 100:.2f}%")

    # ── Save timing ───────────────────────────────────────────────────────────
    with open("timing_transformer.txt", "w") as fh:
        fh.write(f"load,{load_time}\n")
        fh.write(f"train,{train_time}\n")
        fh.write(f"eval,{eval_time}\n")
        fh.write(f"total,{total_time}\n")
        fh.write(f"accuracy,{accuracy}\n")
        fh.write(f"deathrate,{death_rate}\n")
        fh.write(f"precision,{precision}\n")
        fh.write(f"recall,{recall}\n")
        fh.write(f"f1,{f1}\n")

    # ── Save model weights ────────────────────────────────────────────────────
    torch.save(model.state_dict(), "transformer_model.pt")
    print("\nModel weights saved to transformer_model.pt")
    print("Timing results saved to timing_transformer.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
