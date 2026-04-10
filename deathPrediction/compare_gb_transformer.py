#!/usr/bin/env python3
"""
Performance comparison: Gradient Boosting (Serial & OpenMP) vs Transformer
===========================================================================
Reads timing files produced by each implementation and generates comparative
plots covering execution time, speedup, and ML quality metrics.

Timing files consumed:
    timing_gb_serial.txt   – Serial Gradient Boosting (C++)
    timing_gb_openmp.txt   – OpenMP Gradient Boosting (C++)
    timing_transformer.txt – Transformer (Python/PyTorch)

If a timing file is missing the script falls back to representative demo values
so graphs can still be produced for illustration purposes.

Usage:
    python compare_gb_transformer.py [--output-dir <dir>]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

# ---------------------------------------------------------------------------
# Default / demo values used when timing files are absent
# ---------------------------------------------------------------------------
DEMO_DATA = {
    "gb_serial": {
        "load": 0.0142,
        "train": 0.0868,
        "eval": 0.0004,
        "total": 0.1018,
        "accuracy": 1.0000,
        "deathrate": 0.0867,
        "precision": None,
        "recall": None,
        "f1": None,
    },
    "gb_openmp": {
        "load": 0.0143,
        "train": 0.0620,
        "eval": 0.0001,
        "total": 0.0776,
        "accuracy": 1.0000,
        "deathrate": 0.1045,
        "precision": None,
        "recall": None,
        "f1": None,
    },
    "transformer": {
        "load": 0.0150,
        "train": 3.2100,
        "eval": 0.0420,
        "total": 3.2670,
        "accuracy": 0.9213,
        "deathrate": 0.1082,
        "precision": 0.6540,
        "recall": 0.4870,
        "f1": 0.5580,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_timing_file(path: str) -> dict:
    """Parse a key,value timing file; return a dict of float values."""
    result = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, _, val = line.partition(",")
            try:
                result[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return result


def load_timings(timing_dir: str) -> dict:
    """Load all three timing files, falling back to demo data if absent."""
    files = {
        "gb_serial": "timing_gb_serial.txt",
        "gb_openmp": "timing_gb_openmp.txt",
        "transformer": "timing_transformer.txt",
    }
    data = {}
    for key, fname in files.items():
        full_path = os.path.join(timing_dir, fname)
        if os.path.exists(full_path):
            parsed = parse_timing_file(full_path)
            entry = dict(DEMO_DATA[key])   # start from defaults
            entry.update(parsed)           # override with real values
            data[key] = entry
            print(f"  ✓ Loaded {fname}")
        else:
            data[key] = dict(DEMO_DATA[key])
            print(f"  ⚠ {fname} not found – using demo values")
    return data


def save_fig(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ Saved {name}")


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

IMPL_LABELS = [
    "GB Serial\n(C++)",
    "GB OpenMP\n(C++)",
    "Transformer\n(PyTorch)",
]
COLORS = ["#2ecc71", "#3498db", "#e74c3c"]


def fig_total_time(data: dict, out_dir: str):
    """Bar chart of total execution time."""
    keys = ["gb_serial", "gb_openmp", "transformer"]
    times = [data[k]["total"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(IMPL_LABELS, times, color=COLORS, edgecolor="black", linewidth=1.1)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.4f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_ylabel("Total Execution Time (seconds)", fontweight="bold")
    ax.set_title("Total Execution Time Comparison\nGradient Boosting vs Transformer", fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp1_total_execution_time.jpg")


def fig_time_breakdown(data: dict, out_dir: str):
    """Stacked bar chart showing load / train / eval phases."""
    keys = ["gb_serial", "gb_openmp", "transformer"]
    load  = [data[k]["load"]  for k in keys]
    train = [data[k]["train"] for k in keys]
    eval_ = [data[k]["eval"]  for k in keys]
    x = np.arange(len(keys))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    p1 = ax.bar(x, load,  width, label="Load Time",  color="#3498db", edgecolor="black")
    p2 = ax.bar(x, train, width, bottom=load,
                label="Training Time", color="#e74c3c", edgecolor="black")
    p3 = ax.bar(x, eval_,  width,
                bottom=np.array(load) + np.array(train),
                label="Evaluation Time", color="#2ecc71", edgecolor="black")

    for i, (l, tr, ev) in enumerate(zip(load, train, eval_)):
        total = l + tr + ev
        ax.text(i, total, f"{total:.4f}s", ha="center", va="bottom",
                fontweight="bold", fontsize=9)

    ax.set_ylabel("Time (seconds)", fontweight="bold")
    ax.set_title("Execution Time Breakdown by Phase", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(IMPL_LABELS)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp2_time_breakdown.jpg")


def fig_training_time(data: dict, out_dir: str):
    """Bar chart focused on training time (dominant phase)."""
    keys = ["gb_serial", "gb_openmp", "transformer"]
    times = [data[k]["train"] for k in keys]
    baseline = times[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(IMPL_LABELS, times, color=COLORS, edgecolor="black", linewidth=1.1)

    for i, (bar, t) in enumerate(zip(bars, times)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.4f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
        if i > 0:
            ratio = t / baseline
            label = f"{ratio:.2f}× as long"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.set_ylabel("Training Time (seconds)", fontweight="bold")
    ax.set_title(
        "Training Time Comparison\n(Most Computationally Intensive Phase)",
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp3_training_time.jpg")


def fig_speedup(data: dict, out_dir: str):
    """Horizontal bar chart – speedup relative to serial GB baseline."""
    keys = ["gb_serial", "gb_openmp", "transformer"]
    baseline = data["gb_serial"]["total"]
    speedups = [baseline / data[k]["total"] for k in keys]
    bar_colors = ["#2ecc71" if s >= 1 else "#e74c3c" for s in speedups]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(IMPL_LABELS, speedups, color=bar_colors, edgecolor="black",
                   linewidth=1.1, alpha=0.85)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.8,
               label="Serial GB Baseline (1.0×)")

    for bar, s in zip(bars, speedups):
        w = bar.get_width()
        ax.text(
            w + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{s:.2f}×",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xlabel("Speedup Factor  (higher is better)", fontweight="bold")
    ax.set_title(
        "Speedup vs Serial Gradient Boosting Baseline\n(Values < 1.0 indicate slowdown)",
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(speedups) * 1.25)
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp4_speedup.jpg")


def fig_accuracy(data: dict, out_dir: str):
    """Bar chart of prediction accuracy for each implementation."""
    keys = ["gb_serial", "gb_openmp", "transformer"]
    accs = [data[k]["accuracy"] * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(IMPL_LABELS, accs, color=COLORS, edgecolor="black", linewidth=1.1)

    for bar, a in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{a:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Prediction Accuracy Comparison", fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp5_accuracy.jpg")


def fig_quality_metrics(data: dict, out_dir: str):
    """Grouped bar chart of ML quality metrics (accuracy, precision, recall, F1).

    Gradient Boosting implementations do not expose precision/recall/F1 directly,
    so only accuracy is shown for them; the Transformer exposes all four.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    keys = ["gb_serial", "gb_openmp", "transformer"]
    n_impl = len(keys)
    n_met = len(metrics)
    x = np.arange(n_met)
    width = 0.22
    offsets = np.linspace(-(n_impl - 1) / 2, (n_impl - 1) / 2, n_impl) * width

    def _val(key, metric_key):
        v = data[key].get(metric_key)
        return (v * 100) if v is not None else None

    metric_keys = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, label, color) in enumerate(zip(keys, IMPL_LABELS, COLORS)):
        vals = [_val(key, mk) for mk in metric_keys]
        # Replace None with 0 for bar height; annotate N/A separately
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(
            x + offsets[i],
            heights,
            width,
            label=label.replace("\n", " "),
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v is None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    2,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.5,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title(
        "ML Quality Metrics Comparison\n(N/A = metric not reported by that implementation)",
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp6_quality_metrics.jpg")


def fig_comprehensive(data: dict, out_dir: str):
    """2×3 multi-panel summary figure."""
    keys = ["gb_serial", "gb_openmp", "transformer"]
    baseline_total = data["gb_serial"]["total"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ── Panel (0,0): Total time ──────────────────────────────────────────────
    ax = axes[0, 0]
    times = [data[k]["total"] for k in keys]
    bars = ax.bar(range(3), times, color=COLORS, edgecolor="black")
    ax.set_title("Total Execution Time", fontweight="bold")
    ax.set_ylabel("Time (s)", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(IMPL_LABELS, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(times):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # ── Panel (0,1): Training time ───────────────────────────────────────────
    ax = axes[0, 1]
    ttimes = [data[k]["train"] for k in keys]
    ax.bar(range(3), ttimes, color=COLORS, edgecolor="black")
    ax.set_title("Training Time", fontweight="bold")
    ax.set_ylabel("Time (s)", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(IMPL_LABELS, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(ttimes):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # ── Panel (0,2): Speedup ─────────────────────────────────────────────────
    ax = axes[0, 2]
    speedups = [baseline_total / data[k]["total"] for k in keys]
    sc = ["#2ecc71" if s >= 1 else "#e74c3c" for s in speedups]
    ax.bar(range(3), speedups, color=sc, edgecolor="black", alpha=0.85)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Speedup vs Serial GB", fontweight="bold")
    ax.set_ylabel("Speedup Factor", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(IMPL_LABELS, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(speedups):
        ax.text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontsize=8)

    # ── Panel (1,0): Accuracy ────────────────────────────────────────────────
    ax = axes[1, 0]
    accs = [data[k]["accuracy"] * 100 for k in keys]
    ax.bar(range(3), accs, color=COLORS, edgecolor="black")
    ax.set_ylim(0, 115)
    ax.set_title("Prediction Accuracy", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(IMPL_LABELS, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)

    # ── Panel (1,1): Transformer quality metrics ──────────────────────────────
    ax = axes[1, 1]
    t_data = data["transformer"]
    t_metric_defs = [
        ("Accuracy",  t_data.get("accuracy")),
        ("Precision", t_data.get("precision")),
        ("Recall",    t_data.get("recall")),
        ("F1 Score",  t_data.get("f1")),
    ]
    # Only plot metrics that are available (not None)
    available = [(lbl, v * 100) for lbl, v in t_metric_defs if v is not None]
    t_labels_plot, t_vals_plot = zip(*available) if available else ([], [])
    t_colors_all = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    t_colors_plot = t_colors_all[: len(t_vals_plot)]
    if t_vals_plot:
        ax.bar(t_labels_plot, t_vals_plot, color=t_colors_plot, edgecolor="black")
        ax.set_ylim(0, 115)
        for i, v in enumerate(t_vals_plot):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("Transformer Quality Metrics", fontweight="bold")
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel (1,2): Time-breakdown stacked ──────────────────────────────────
    ax = axes[1, 2]
    load  = [data[k]["load"]  for k in keys]
    train = [data[k]["train"] for k in keys]
    eval_ = [data[k]["eval"]  for k in keys]
    x = np.arange(3)
    w = 0.5
    ax.bar(x, load,  w, label="Load",  color="#3498db", edgecolor="black")
    ax.bar(x, train, w, bottom=load, label="Train", color="#e74c3c", edgecolor="black")
    ax.bar(x, eval_, w, bottom=np.array(load) + np.array(train),
           label="Eval", color="#2ecc71", edgecolor="black")
    ax.set_title("Time Breakdown by Phase", fontweight="bold")
    ax.set_ylabel("Time (s)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(IMPL_LABELS, rotation=20, ha="right")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Comprehensive Performance Comparison\nGradient Boosting (Serial & OpenMP) vs Transformer",
        fontweight="bold",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    save_fig(fig, out_dir, "cmp7_comprehensive.jpg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate GB vs Transformer performance comparison plots"
    )
    parser.add_argument(
        "--timing-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing timing_*.txt files (default: script directory)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write output JPG files (default: same as --timing-dir)",
    )
    args = parser.parse_args()

    timing_dir = args.timing_dir
    output_dir = args.output_dir or timing_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Gradient Boosting vs Transformer – Performance Comparison")
    print("=" * 60)
    print("\nLoading timing data …")
    data = load_timings(timing_dir)

    print("\nGenerating plots …")
    fig_total_time(data, output_dir)
    fig_time_breakdown(data, output_dir)
    fig_training_time(data, output_dir)
    fig_speedup(data, output_dir)
    fig_accuracy(data, output_dir)
    fig_quality_metrics(data, output_dir)
    fig_comprehensive(data, output_dir)

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print("\nGenerated files (in", output_dir, "):")
    files = [
        ("cmp1_total_execution_time.jpg", "Overall execution time comparison"),
        ("cmp2_time_breakdown.jpg",       "Stacked time breakdown by phase"),
        ("cmp3_training_time.jpg",        "Training phase comparison"),
        ("cmp4_speedup.jpg",              "Speedup vs serial GB baseline"),
        ("cmp5_accuracy.jpg",             "Prediction accuracy comparison"),
        ("cmp6_quality_metrics.jpg",      "Full ML quality metrics (Acc / Prec / Rec / F1)"),
        ("cmp7_comprehensive.jpg",        "Multi-panel comprehensive summary"),
    ]
    for fname, desc in files:
        print(f"  {fname}  –  {desc}")
    print()


if __name__ == "__main__":
    main()
