# plot_performance.py

import os
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_timing_breakdown(lines):
    timing = {}
    start = False
    for line in lines:
        if line.strip().startswith("Implementation"):
            start = True
            continue
        if start:
            if line.strip() == "" or line.startswith("-"):
                continue
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) == 5:
                impl, load, train, eval_, total = parts
                try:
                    timing[impl] = {
                        'Load': float(load),
                        'Train': float(train),
                        'Eval': float(eval_),
                        'Total': float(total)
                    }
                except ValueError:
                    continue
            elif line.startswith("-"):
                break
    return timing

def parse_parallel_metrics(lines):
    metrics = {}
    start = False
    for line in lines:
        if line.strip().startswith("Implementation"):
            start = True
            continue
        if start:
            if line.strip() == "" or line.startswith("-"):
                continue
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 4:
                impl = parts[0]
                try:
                    threads = int(parts[1])
                    speedup = float(parts[2])
                    efficiency = float(parts[3])
                    metrics[impl] = {
                        'Threads': threads,
                        'Speedup': speedup,
                        'Efficiency': efficiency
                    }
                except ValueError:
                    continue
            elif line.startswith("-"):
                break
    return metrics

def parse_model_performance(lines):
    perf = {}
    start = False
    for line in lines:
        if line.strip().startswith("Implementation"):
            start = True
            continue
        if start:
            if line.strip() == "" or line.startswith("-"):
                continue
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) == 3:
                impl = parts[0]
                try:
                    accuracy = float(parts[1])
                    death_rate = float(parts[2])
                    perf[impl] = {
                        'Accuracy': accuracy,
                        'Death Rate': death_rate
                    }
                except ValueError:
                    continue
            elif line.startswith("-"):
                break
    return perf

def extract_sections(report_path):
    with open(report_path) as f:
        lines = f.readlines()

    timing_section = []
    parallel_section = []
    model_section = []
    in_timing = in_parallel = in_model = False
    for line in lines:
        if "TIMING BREAKDOWN" in line:
            in_timing = True
            continue
        if "PARALLEL PERFORMANCE METRICS" in line:
            in_timing = False
            in_parallel = True
            continue
        if "MODEL PERFORMANCE" in line:
            in_parallel = False
            in_model = True
            continue
        if "KEY FINDINGS" in line:
            in_model = False
        if in_timing:
            timing_section.append(line)
        if in_parallel:
            parallel_section.append(line)
        if in_model:
            model_section.append(line)

    timing = parse_timing_breakdown(timing_section)
    parallel = parse_parallel_metrics(parallel_section)
    model_perf = parse_model_performance(model_section)
    return timing, parallel, model_perf

def generate_plots(timing, parallel, model_perf, prefix=''):
    implementations = list(timing.keys())

    load   = [timing[impl]['Load']  for impl in implementations]
    train  = [timing[impl]['Train'] for impl in implementations]
    eval_  = [timing[impl]['Eval']  for impl in implementations]
    totals = [timing[impl]['Total'] for impl in implementations]

    ind = np.arange(len(implementations))
    width = 0.6

    # --- Stacked bar: Load / Train / Eval ---
    plt.figure(figsize=(max(10, len(implementations) * 1.4), 6))
    p1 = plt.bar(ind, load, width, label='Load')
    p2 = plt.bar(ind, train, width, bottom=load, label='Train')
    bottom2 = [l + t for l, t in zip(load, train)]
    p3 = plt.bar(ind, eval_, width, bottom=bottom2, label='Eval')
    plt.ylabel('Time (s)')
    plt.title(f'{prefix}Timing Breakdown by Implementation')
    plt.xticks(ind, implementations, rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    out = f'{prefix.lower().replace(" ", "_")}timing_breakdown_stacked.png'
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # --- Total time bar ---
    plt.figure(figsize=(max(8, len(implementations) * 1.4), 6))
    bars_t = plt.bar(implementations, totals, color='steelblue', edgecolor='black')
    min_idx = totals.index(min(totals))
    bars_t[min_idx].set_edgecolor('gold')
    bars_t[min_idx].set_linewidth(3)
    for i, v in enumerate(totals):
        plt.text(i, v, f'{v:.4f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.ylabel('Total Time (s)')
    plt.title(f'{prefix}Total Execution Time Comparison')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    out = f'{prefix.lower().replace(" ", "_")}total_time.png'
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # --- Speedup bar ---
    serial_time = totals[0]
    speedups = [serial_time / t for t in totals]
    spd_colors = ['green' if s >= 1 else 'red' for s in speedups]
    plt.figure(figsize=(max(8, len(implementations) * 1.4), 6))
    plt.bar(implementations, speedups, color=spd_colors, edgecolor='black', alpha=0.8)
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    for i, v in enumerate(speedups):
        plt.text(i, v, f'{v:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.ylabel('Speedup')
    plt.title(f'{prefix}Speedup vs Serial Baseline')
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    out = f'{prefix.lower().replace(" ", "_")}speedup.png'
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # --- Parallel efficiency ---
    if parallel:
        eff      = [parallel[impl]['Efficiency'] for impl in implementations if impl in parallel]
        impls_eff = [impl for impl in implementations if impl in parallel]
        plt.figure(figsize=(max(8, len(impls_eff) * 1.4), 6))
        plt.bar(impls_eff, eff, color='orange', edgecolor='black')
        plt.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Ideal 100%')
        plt.ylabel('Efficiency (%)')
        plt.title(f'{prefix}Parallel Efficiency by Implementation')
        plt.xticks(rotation=30, ha='right')
        plt.legend()
        plt.tight_layout()
        out = f'{prefix.lower().replace(" ", "_")}parallel_efficiency.png'
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")

        threads = [parallel[impl]['Threads'] for impl in implementations if impl in parallel]
        plt.figure(figsize=(max(8, len(impls_eff) * 1.4), 6))
        plt.bar(impls_eff, threads, color='purple', edgecolor='black')
        plt.ylabel('Threads')
        plt.title(f'{prefix}Threads by Implementation')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out = f'{prefix.lower().replace(" ", "_")}threads.png'
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")

    # --- Accuracy and death rate ---
    if model_perf:
        acc = [model_perf[impl]['Accuracy']   for impl in implementations if impl in model_perf]
        dr  = [model_perf[impl]['Death Rate'] for impl in implementations if impl in model_perf]
        impls_acc = [impl for impl in implementations if impl in model_perf]

        plt.figure(figsize=(max(8, len(impls_acc) * 1.4), 6))
        plt.bar(impls_acc, acc, color='teal', edgecolor='black')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{prefix}Accuracy by Implementation')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out = f'{prefix.lower().replace(" ", "_")}accuracy.png'
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")

        plt.figure(figsize=(max(8, len(impls_acc) * 1.4), 6))
        plt.bar(impls_acc, dr, color='red', edgecolor='black')
        plt.ylabel('Death Rate (%)')
        plt.title(f'{prefix}Death Rate by Implementation')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out = f'{prefix.lower().replace(" ", "_")}death_rate.png'
        plt.savefig(out)
        plt.close()
        print(f"Saved: {out}")

def main():
    # -----------------------------------------------------------
    # Logistic Regression report
    # -----------------------------------------------------------
    if os.path.exists('performance_report.txt'):
        print("\n=== Logistic Regression (performance_report.txt) ===")
        timing, parallel, model_perf = extract_sections('performance_report.txt')
        generate_plots(timing, parallel, model_perf, prefix='LR ')
    else:
        print("performance_report.txt not found – skipping LR plots.")

    # -----------------------------------------------------------
    # Random Forest report
    # -----------------------------------------------------------
    if os.path.exists('rf_performance_report.txt'):
        print("\n=== Random Forest (rf_performance_report.txt) ===")
        rf_timing, rf_parallel, rf_model_perf = extract_sections('rf_performance_report.txt')
        generate_plots(rf_timing, rf_parallel, rf_model_perf, prefix='RF ')
    else:
        print("rf_performance_report.txt not found – skipping RF plots.")
        print("Run the comparison_runner first: ./comparison_runner mimic_data.csv 4")

if __name__ == "__main__":
    main()

