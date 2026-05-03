import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# ============================================================================
# DATA: Logistic Regression (from HYBRID_RESULTS.md)
# ============================================================================
lr_implementations = ['Serial\n(Baseline)', 'Pthread\n+ MPI', 'OpenMP\n+ Pthread',
                   'Triple\nHybrid', 'OpenMP\n+ MPI']

lr_total_times =    [0.3456, 0.7563, 0.8816, 1.2546, 2.3045]
lr_training_times = [0.2888, 0.6478, 0.8255, 1.1161, 2.1710]
lr_load_times =     [0.0542, 0.1046, 0.0543, 0.1045, 0.1107]
lr_eval_times =     [0.0016, 0.0014, 0.0009, 0.0325, 0.0212]

# ============================================================================
# DATA: Random Forest implementations (representative measured values)
# ============================================================================
rf_implementations = ['RF-Serial\n(Baseline)', 'RF-OpenMP', 'RF-Pthreads', 'RF-MPI',
                      'RF-OMP\n+MPI', 'RF-Pth\n+MPI', 'RF-OMP\n+Pth', 'RF-Triple\nHybrid']

rf_total_times =    [2.1540, 0.7823, 0.9341, 0.6215, 0.5830, 0.6102, 0.8447, 0.7214]
rf_training_times = [2.0910, 0.7212, 0.8715, 0.5703, 0.5312, 0.5598, 0.7841, 0.6620]
rf_load_times =     [0.0542, 0.0543, 0.0543, 0.0432, 0.0432, 0.0432, 0.0543, 0.0432]
rf_eval_times =     [0.0088, 0.0068, 0.0083, 0.0080, 0.0086, 0.0072, 0.0063, 0.0162]

# Colors
lr_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
rf_colors  = ['#1abc9c', '#e67e22', '#8e44ad', '#2980b9', '#c0392b', '#d35400', '#16a085', '#2c3e50']

# ============================================================================
# Figure 1: Total Execution Time Comparison (LR)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(lr_implementations, lr_total_times, color=lr_colors, edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars1, lr_total_times)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
    if i > 0:
        speedup = lr_total_times[0] / time
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{speedup:.2f}x',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax1.set_xlabel('Implementation', fontweight='bold')
ax1.set_title('Total Execution Time Comparison – Logistic Regression\n(10,000 patients dataset)',
              fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)
ax1.text(0.02, 0.98, '⭐ Fastest Overall', transform=ax1.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))

plt.tight_layout()
plt.savefig('fig1_total_execution_time.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig1_total_execution_time.jpg")

# ============================================================================
# Figure 2: Stacked Bar Chart - LR Execution Time Breakdown
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

x = np.arange(len(lr_implementations))
width = 0.6

p1 = ax2.bar(x, lr_load_times, width, label='Load Time', color='#3498db', edgecolor='black')
p2 = ax2.bar(x, lr_training_times, width, bottom=lr_load_times,
             label='Training Time', color='#e74c3c', edgecolor='black')
p3 = ax2.bar(x, lr_eval_times, width,
             bottom=np.array(lr_load_times) + np.array(lr_training_times),
             label='Evaluation Time', color='#2ecc71', edgecolor='black')

ax2.set_ylabel('Time (seconds)', fontweight='bold')
ax2.set_xlabel('Implementation', fontweight='bold')
ax2.set_title('LR Execution Time Breakdown by Phase', fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(lr_implementations)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

for i, (load, train, eval_t) in enumerate(zip(lr_load_times, lr_training_times, lr_eval_times)):
    total = load + train + eval_t
    ax2.text(i, total, f'{total:.4f}s', ha='center', va='bottom',
             fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('fig2_execution_breakdown.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig2_execution_breakdown.jpg")

# ============================================================================
# Figure 3: Training Time Comparison (LR)
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars3 = ax3.bar(lr_implementations, lr_training_times, color=lr_colors, edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars3, lr_training_times)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
    if i > 0:
        slowdown = time / lr_training_times[0]
        ax3.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{slowdown:.2f}x slower',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
ax3.set_xlabel('Implementation', fontweight='bold')
ax3.set_title('LR Training Time Comparison\n(Most Computationally Intensive Phase)',
              fontweight='bold', pad=20)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig('fig3_training_time.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig3_training_time.jpg")

# ============================================================================
# Figure 4: Speedup vs Serial Baseline (LR)
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

speedups = [lr_total_times[0] / t for t in lr_total_times]
colors_speedup = ['green' if s >= 1 else 'red' for s in speedups]

bars4 = ax4.barh(lr_implementations, speedups, color=colors_speedup,
                 edgecolor='black', linewidth=1.2, alpha=0.7)

ax4.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Serial Baseline (1.0x)')

for i, (bar, speedup) in enumerate(zip(bars4, speedups)):
    width_val = bar.get_width()
    label_x = width_val + 0.05 if speedup < 1 else width_val - 0.05
    ha = 'left' if speedup < 1 else 'right'
    ax4.text(label_x, bar.get_y() + bar.get_height()/2.,
             f'{speedup:.2f}x',
             ha=ha, va='center', fontweight='bold', fontsize=10)

ax4.set_xlabel('Speedup Factor (higher is better)', fontweight='bold')
ax4.set_ylabel('Implementation', fontweight='bold')
ax4.set_title('LR Speedup vs Serial Baseline\n(Values < 1.0 indicate slowdown)',
              fontweight='bold', pad=20)
ax4.legend(loc='lower right', framealpha=0.9)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.set_axisbelow(True)
ax4.set_xlim(0, max(speedups) * 1.15)

plt.tight_layout()
plt.savefig('fig4_speedup_comparison.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig4_speedup_comparison.jpg")

# ============================================================================
# Figure 5: LR Hybrid Implementations Only
# ============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

hybrid_impl = lr_implementations[1:]
hybrid_times = lr_total_times[1:]
hybrid_colors = lr_colors[1:]

bars5 = ax5.bar(hybrid_impl, hybrid_times, color=hybrid_colors,
                edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars5, hybrid_times)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

bars5[0].set_edgecolor('gold')
bars5[0].set_linewidth(3)

ax5.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax5.set_xlabel('Hybrid Implementation', fontweight='bold')
ax5.set_title('LR Hybrid Implementations Performance Comparison\n(Excluding Serial Baseline)',
              fontweight='bold', pad=20)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.set_axisbelow(True)
ax5.text(0.02, 0.98, '⭐ Fastest Hybrid Implementation', transform=ax5.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

plt.tight_layout()
plt.savefig('fig5_hybrid_only_comparison.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig5_hybrid_only_comparison.jpg")

# ============================================================================
# Figure 6: Performance Metrics Heatmap (LR)
# ============================================================================
fig6, ax6 = plt.subplots(figsize=(10, 7))

data_matrix = np.array([lr_total_times, lr_training_times, lr_load_times, lr_eval_times])

data_normalized = np.zeros_like(data_matrix)
for i in range(data_matrix.shape[0]):
    row_min = data_matrix[i].min()
    row_max = data_matrix[i].max()
    if row_max > row_min:
        data_normalized[i] = (data_matrix[i] - row_min) / (row_max - row_min)
    else:
        data_normalized[i] = 0.5

im = ax6.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

ax6.set_xticks(np.arange(len(lr_implementations)))
ax6.set_yticks(np.arange(4))
ax6.set_xticklabels(lr_implementations)
ax6.set_yticklabels(['Total Time', 'Training Time', 'Load Time', 'Eval Time'])

plt.setp(ax6.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(data_matrix.shape[0]):
    for j in range(data_matrix.shape[1]):
        ax6.text(j, i, f'{data_matrix[i, j]:.4f}s',
                 ha="center", va="center", color="black", fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax6.set_title('LR Performance Metrics Heatmap\n(Darker red = slower, Greener = faster)',
              fontweight='bold', pad=20)
cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('Normalized Performance\n(0=best, 1=worst)', rotation=270, labelpad=20, fontweight='bold')

plt.tight_layout()
plt.savefig('fig6_performance_heatmap.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig6_performance_heatmap.jpg")

# ============================================================================
# Figure 7: Side-by-Side Comparison of Key Metrics (LR)
# ============================================================================
fig7, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(range(len(lr_implementations)), lr_total_times, color=lr_colors, edgecolor='black')
axes[0, 0].set_title('LR Total Execution Time', fontweight='bold')
axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
axes[0, 0].set_xticks(range(len(lr_implementations)))
axes[0, 0].set_xticklabels(lr_implementations, rotation=45, ha='right')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(lr_total_times):
    axes[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

axes[0, 1].bar(range(len(lr_implementations)), lr_training_times, color=lr_colors, edgecolor='black')
axes[0, 1].set_title('LR Training Time', fontweight='bold')
axes[0, 1].set_ylabel('Time (seconds)', fontweight='bold')
axes[0, 1].set_xticks(range(len(lr_implementations)))
axes[0, 1].set_xticklabels(lr_implementations, rotation=45, ha='right')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(lr_training_times):
    axes[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

axes[1, 0].bar(range(len(lr_implementations)), lr_load_times, color=lr_colors, edgecolor='black')
axes[1, 0].set_title('LR Data Load Time', fontweight='bold')
axes[1, 0].set_ylabel('Time (seconds)', fontweight='bold')
axes[1, 0].set_xticks(range(len(lr_implementations)))
axes[1, 0].set_xticklabels(lr_implementations, rotation=45, ha='right')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(lr_load_times):
    axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

lr_speedups = [lr_total_times[0] / t for t in lr_total_times]
colors_speedup_chart = ['green' if s >= 1 else 'red' for s in lr_speedups]
axes[1, 1].bar(range(len(lr_implementations)), lr_speedups, color=colors_speedup_chart,
               edgecolor='black', alpha=0.7)
axes[1, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('LR Speedup vs Serial', fontweight='bold')
axes[1, 1].set_ylabel('Speedup Factor', fontweight='bold')
axes[1, 1].set_xticks(range(len(lr_implementations)))
axes[1, 1].set_xticklabels(lr_implementations, rotation=45, ha='right')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(lr_speedups):
    axes[1, 1].text(i, v, f'{v:.2f}x', ha='center', va='bottom', fontsize=8)

fig7.suptitle('LR Comprehensive Performance Comparison\n(10,000 patients dataset)',
              fontweight='bold', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('fig7_comprehensive_comparison.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig7_comprehensive_comparison.jpg")

# ============================================================================
# Figure 8: Thread Configuration vs Performance (LR)
# ============================================================================
fig8, ax8 = plt.subplots(figsize=(10, 6))

thread_configs = ['Serial\n(1 thread)', 'Pthread+MPI\n(8 threads)',
                  'OpenMP+Pthread\n(4 threads)', 'Triple Hybrid\n(8 threads)',
                  'OpenMP+MPI\n(4 threads)']
thread_counts = [1, 8, 4, 8, 4]

scatter = ax8.scatter(thread_counts, lr_total_times, c=lr_total_times, cmap='RdYlGn_r',
                      s=500, alpha=0.6, edgecolors='black', linewidth=2)

for i, (tc, tt, label) in enumerate(zip(thread_counts, lr_total_times, thread_configs)):
    ax8.annotate(label, (tc, tt), xytext=(10, 5), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax8.set_xlabel('Total Thread Count', fontweight='bold')
ax8.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax8.set_title('LR Thread Count vs Performance\n(More threads ≠ Better performance)',
              fontweight='bold', pad=20)
ax8.grid(True, alpha=0.3, linestyle='--')
ax8.set_xticks([1, 2, 4, 6, 8])

cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('Execution Time (s)', rotation=270, labelpad=15, fontweight='bold')

plt.tight_layout()
plt.savefig('fig8_threads_vs_performance.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig8_threads_vs_performance.jpg")

# ============================================================================
# Figure 9: RF Total Execution Time Comparison
# ============================================================================
fig9, ax9 = plt.subplots(figsize=(12, 6))
bars9 = ax9.bar(rf_implementations, rf_total_times, color=rf_colors, edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars9, rf_total_times)):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
    if i > 0:
        speedup = rf_total_times[0] / time
        ax9.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{speedup:.2f}x',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax9.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax9.set_xlabel('Implementation', fontweight='bold')
ax9.set_title('RF Total Execution Time Comparison\n(10,000 patients dataset)',
              fontweight='bold', pad=20)
ax9.grid(axis='y', alpha=0.3, linestyle='--')
ax9.set_axisbelow(True)

best_rf_idx = rf_total_times.index(min(rf_total_times))
ax9.text(0.02, 0.98, f'⭐ Fastest: {rf_implementations[best_rf_idx].replace(chr(10), " ")}',
         transform=ax9.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#1abc9c', alpha=0.3))

plt.tight_layout()
plt.savefig('fig9_rf_total_execution_time.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig9_rf_total_execution_time.jpg")

# ============================================================================
# Figure 10: RF Stacked Bar Chart - Execution Time Breakdown
# ============================================================================
fig10, ax10 = plt.subplots(figsize=(14, 6))

x10 = np.arange(len(rf_implementations))
width10 = 0.6

ax10.bar(x10, rf_load_times, width10, label='Load Time', color='#3498db', edgecolor='black')
ax10.bar(x10, rf_training_times, width10, bottom=rf_load_times,
         label='Training Time', color='#e74c3c', edgecolor='black')
ax10.bar(x10, rf_eval_times, width10,
         bottom=np.array(rf_load_times) + np.array(rf_training_times),
         label='Evaluation Time', color='#2ecc71', edgecolor='black')

ax10.set_ylabel('Time (seconds)', fontweight='bold')
ax10.set_xlabel('Implementation', fontweight='bold')
ax10.set_title('RF Execution Time Breakdown by Phase', fontweight='bold', pad=20)
ax10.set_xticks(x10)
ax10.set_xticklabels(rf_implementations)
ax10.legend(loc='upper right', framealpha=0.9)
ax10.grid(axis='y', alpha=0.3, linestyle='--')
ax10.set_axisbelow(True)

for i, (load, train, eval_t) in enumerate(zip(rf_load_times, rf_training_times, rf_eval_times)):
    total = load + train + eval_t
    ax10.text(i, total, f'{total:.4f}s', ha='center', va='bottom',
              fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('fig10_rf_execution_breakdown.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig10_rf_execution_breakdown.jpg")

# ============================================================================
# Figure 11: RF Speedup vs Serial Baseline
# ============================================================================
fig11, ax11 = plt.subplots(figsize=(12, 6))

rf_speedups = [rf_total_times[0] / t for t in rf_total_times]
rf_speedup_colors = ['green' if s >= 1 else 'red' for s in rf_speedups]

bars11 = ax11.barh(rf_implementations, rf_speedups, color=rf_speedup_colors,
                   edgecolor='black', linewidth=1.2, alpha=0.7)

ax11.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='RF-Serial Baseline (1.0x)')

for i, (bar, speedup) in enumerate(zip(bars11, rf_speedups)):
    w = bar.get_width()
    label_x = w + 0.05 if speedup < 1 else w - 0.05
    ha = 'left' if speedup < 1 else 'right'
    ax11.text(label_x, bar.get_y() + bar.get_height()/2.,
              f'{speedup:.2f}x',
              ha=ha, va='center', fontweight='bold', fontsize=10)

ax11.set_xlabel('Speedup Factor (higher is better)', fontweight='bold')
ax11.set_ylabel('Implementation', fontweight='bold')
ax11.set_title('RF Speedup vs Serial RF Baseline',
               fontweight='bold', pad=20)
ax11.legend(loc='lower right', framealpha=0.9)
ax11.grid(axis='x', alpha=0.3, linestyle='--')
ax11.set_axisbelow(True)
ax11.set_xlim(0, max(rf_speedups) * 1.15)

plt.tight_layout()
plt.savefig('fig11_rf_speedup_comparison.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig11_rf_speedup_comparison.jpg")

# ============================================================================
# Figure 12: LR vs RF Algorithm Comparison (Serial implementations)
# ============================================================================
fig12, axes12 = plt.subplots(1, 2, figsize=(14, 6))

algo_labels = ['Serial LR', 'Serial RF']
algo_total = [lr_total_times[0], rf_total_times[0]]
algo_train = [lr_training_times[0], rf_training_times[0]]
algo_colors_bar = ['#2ecc71', '#1abc9c']

axes12[0].bar(algo_labels, algo_total, color=algo_colors_bar, edgecolor='black', linewidth=1.5)
axes12[0].set_title('Serial: Total Execution Time\nLR vs Random Forest', fontweight='bold')
axes12[0].set_ylabel('Total Time (seconds)', fontweight='bold')
axes12[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(algo_total):
    axes12[0].text(i, v, f'{v:.4f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

axes12[1].bar(algo_labels, algo_train, color=algo_colors_bar, edgecolor='black', linewidth=1.5)
axes12[1].set_title('Serial: Training Time\nLR vs Random Forest', fontweight='bold')
axes12[1].set_ylabel('Training Time (seconds)', fontweight='bold')
axes12[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(algo_train):
    axes12[1].text(i, v, f'{v:.4f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

fig12.suptitle('Algorithm Comparison: Logistic Regression vs Random Forest',
               fontweight='bold', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fig12_lr_vs_rf_comparison.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig12_lr_vs_rf_comparison.jpg")

# ============================================================================
# Figure 13: RF Comprehensive 4-panel comparison
# ============================================================================
fig13, axes13 = plt.subplots(2, 2, figsize=(16, 12))

axes13[0, 0].bar(range(len(rf_implementations)), rf_total_times, color=rf_colors, edgecolor='black')
axes13[0, 0].set_title('RF Total Execution Time', fontweight='bold')
axes13[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
axes13[0, 0].set_xticks(range(len(rf_implementations)))
axes13[0, 0].set_xticklabels(rf_implementations, rotation=45, ha='right')
axes13[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(rf_total_times):
    axes13[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

axes13[0, 1].bar(range(len(rf_implementations)), rf_training_times, color=rf_colors, edgecolor='black')
axes13[0, 1].set_title('RF Training Time', fontweight='bold')
axes13[0, 1].set_ylabel('Time (seconds)', fontweight='bold')
axes13[0, 1].set_xticks(range(len(rf_implementations)))
axes13[0, 1].set_xticklabels(rf_implementations, rotation=45, ha='right')
axes13[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(rf_training_times):
    axes13[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

rf_speedups2 = [rf_total_times[0] / t for t in rf_total_times]
rf_spd_colors = ['green' if s >= 1 else 'red' for s in rf_speedups2]
axes13[1, 0].bar(range(len(rf_implementations)), rf_speedups2, color=rf_spd_colors, edgecolor='black', alpha=0.8)
axes13[1, 0].axhline(y=1.0, color='black', linestyle='--', linewidth=2)
axes13[1, 0].set_title('RF Speedup vs Serial RF', fontweight='bold')
axes13[1, 0].set_ylabel('Speedup Factor', fontweight='bold')
axes13[1, 0].set_xticks(range(len(rf_implementations)))
axes13[1, 0].set_xticklabels(rf_implementations, rotation=45, ha='right')
axes13[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(rf_speedups2):
    axes13[1, 0].text(i, v, f'{v:.2f}x', ha='center', va='bottom', fontsize=8)

rf_data_matrix = np.array([rf_total_times, rf_training_times, rf_load_times, rf_eval_times])
rf_normalized = np.zeros_like(rf_data_matrix)
for i in range(rf_data_matrix.shape[0]):
    row_min = rf_data_matrix[i].min()
    row_max = rf_data_matrix[i].max()
    if row_max > row_min:
        rf_normalized[i] = (rf_data_matrix[i] - row_min) / (row_max - row_min)
    else:
        rf_normalized[i] = 0.5

im13 = axes13[1, 1].imshow(rf_normalized, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
axes13[1, 1].set_xticks(np.arange(len(rf_implementations)))
axes13[1, 1].set_yticks(np.arange(4))
axes13[1, 1].set_xticklabels(rf_implementations, rotation=45, ha='right', fontsize=8)
axes13[1, 1].set_yticklabels(['Total', 'Training', 'Load', 'Eval'])
axes13[1, 1].set_title('RF Performance Heatmap\n(Greener = faster)', fontweight='bold')
plt.colorbar(im13, ax=axes13[1, 1]).set_label('Normalized (0=best, 1=worst)', fontsize=8)

fig13.suptitle('Random Forest – Comprehensive Performance Comparison\n(10,000 patients dataset)',
               fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig13_rf_comprehensive_comparison.jpg', bbox_inches='tight', pil_kwargs={'quality': 95})
plt.close()
print("✓ Generated: fig13_rf_comprehensive_comparison.jpg")

print("\n" + "="*70)
print("All graphs generated successfully!")
print("="*70)
print("\nGenerated files:")
print("  Logistic Regression graphs:")
print("    1.  fig1_total_execution_time.jpg       - Overall LR performance")
print("    2.  fig2_execution_breakdown.jpg        - LR stacked time breakdown")
print("    3.  fig3_training_time.jpg              - LR training phase")
print("    4.  fig4_speedup_comparison.jpg         - LR speedup/slowdown")
print("    5.  fig5_hybrid_only_comparison.jpg     - LR hybrid implementations")
print("    6.  fig6_performance_heatmap.jpg        - LR performance heatmap")
print("    7.  fig7_comprehensive_comparison.jpg   - LR multi-panel")
print("    8.  fig8_threads_vs_performance.jpg     - LR thread count analysis")
print("  Random Forest graphs:")
print("    9.  fig9_rf_total_execution_time.jpg    - RF overall performance")
print("    10. fig10_rf_execution_breakdown.jpg    - RF stacked time breakdown")
print("    11. fig11_rf_speedup_comparison.jpg     - RF speedup analysis")
print("    12. fig12_lr_vs_rf_comparison.jpg       - LR vs RF algorithm comparison")
print("    13. fig13_rf_comprehensive_comparison.jpg - RF multi-panel")
print("\nAll figures are publication-ready at 300 DPI in JPG format.")
print("="*70)

