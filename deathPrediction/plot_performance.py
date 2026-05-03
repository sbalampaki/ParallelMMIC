# plot_performance.py

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

def main():
    with open('performance_report.txt') as f:
        lines = f.readlines()

    # Find relevant sections
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

    implementations = list(timing.keys())

    # Stacked bar chart: Load, Train, Eval times
    load = [timing[impl]['Load'] for impl in implementations]
    train = [timing[impl]['Train'] for impl in implementations]
    eval_ = [timing[impl]['Eval'] for impl in implementations]

    ind = np.arange(len(implementations))
    width = 0.6

    plt.figure(figsize=(10,6))
    p1 = plt.bar(ind, load, width, label='Load')
    p2 = plt.bar(ind, train, width, bottom=load, label='Train')
    bottom2 = [l+t for l, t in zip(load, train)]
    p3 = plt.bar(ind, eval_, width, bottom=bottom2, label='Eval')
    plt.ylabel('Time (s)')
    plt.title('Timing Breakdown by Implementation')
    plt.xticks(ind, implementations)
    plt.legend()
    plt.tight_layout()
    plt.savefig('timing_breakdown_stacked.png')
    plt.close()

    # Bar chart: Parallel Efficiency
    if parallel:
        eff = [parallel[impl]['Efficiency'] for impl in implementations if impl in parallel]
        impls_eff = [impl for impl in implementations if impl in parallel]
        plt.figure(figsize=(8,6))
        plt.bar(impls_eff, eff, color='orange')
        plt.ylabel('Efficiency (%)')
        plt.title('Parallel Efficiency by Implementation')
        plt.tight_layout()
        plt.savefig('parallel_efficiency.png')
        plt.close()

        # Bar chart: Threads
        threads = [parallel[impl]['Threads'] for impl in implementations if impl in parallel]
        plt.figure(figsize=(8,6))
        plt.bar(impls_eff, threads, color='purple')
        plt.ylabel('Threads')
        plt.title('Threads by Implementation')
        plt.tight_layout()
        plt.savefig('threads.png')
        plt.close()

    # Bar chart: Accuracy
    if model_perf:
        acc = [model_perf[impl]['Accuracy'] for impl in implementations if impl in model_perf]
        impls_acc = [impl for impl in implementations if impl in model_perf]
        plt.figure(figsize=(8,6))
        plt.bar(impls_acc, acc, color='teal')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy by Implementation')
        plt.tight_layout()
        plt.savefig('accuracy.png')
        plt.close()

        # Bar chart: Death Rate
        dr = [model_perf[impl]['Death Rate'] for impl in implementations if impl in model_perf]
        plt.figure(figsize=(8,6))
        plt.bar(impls_acc, dr, color='red')
        plt.ylabel('Death Rate (%)')
        plt.title('Death Rate by Implementation')
        plt.tight_layout()
        plt.savefig('death_rate.png')
        plt.close()

    print("Saved: timing_breakdown_stacked.png, parallel_efficiency.png, threads.png, accuracy.png, death_rate.png")

if __name__ == "__main__":
    main()
