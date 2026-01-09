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

# Data from HYBRID_RESULTS.md
implementations = ['Serial\n(Baseline)', 'Pthread\n+ MPI', 'OpenMP\n+ Pthread', 
                   'Triple\nHybrid', 'OpenMP\n+ MPI']

# Execution times (seconds)
total_times = [0.3456, 0.7563, 0.8816, 1.2546, 2.3045]
training_times = [0.2888, 0.6478, 0.8255, 1.1161, 2.1710]
load_times = [0.0542, 0.1046, 0.0543, 0.1045, 0.1107]
eval_times = [0.0016, 0.0014, 0.0009, 0.0325, 0.0212]

# Colors
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

# ============================================================================
# Figure 1: Total Execution Time Comparison
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(implementations, total_times, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars1, total_times)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add speedup annotation
    if i > 0:
        speedup = total_times[0] / time
        ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{speedup:.2f}x',
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax1.set_xlabel('Implementation', fontweight='bold')
ax1.set_title('Total Execution Time Comparison\n(10,000 patients dataset)', 
              fontweight='bold', pad=20)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

# Add legend for fastest
ax1.text(0.02, 0.98, '⭐ Fastest Overall', transform=ax1.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))

plt.tight_layout()
plt.savefig('fig1_total_execution_time.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig1_total_execution_time.jpg")

# ============================================================================
# Figure 2: Stacked Bar Chart - Execution Time Breakdown
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

x = np.arange(len(implementations))
width = 0.6

p1 = ax2.bar(x, load_times, width, label='Load Time', color='#3498db', edgecolor='black')
p2 = ax2.bar(x, training_times, width, bottom=load_times, 
             label='Training Time', color='#e74c3c', edgecolor='black')
p3 = ax2.bar(x, eval_times, width, 
             bottom=np.array(load_times) + np.array(training_times),
             label='Evaluation Time', color='#2ecc71', edgecolor='black')

ax2.set_ylabel('Time (seconds)', fontweight='bold')
ax2.set_xlabel('Implementation', fontweight='bold')
ax2.set_title('Execution Time Breakdown by Phase', fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(implementations)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add total time labels on top
for i, (load, train, eval_t) in enumerate(zip(load_times, training_times, eval_times)):
    total = load + train + eval_t
    ax2.text(i, total, f'{total:.4f}s', ha='center', va='bottom', 
             fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('fig2_execution_breakdown.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig2_execution_breakdown.jpg")

# ============================================================================
# Figure 3: Training Time Comparison (Most Critical Phase)
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars3 = ax3.bar(implementations, training_times, color=colors, edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars3, training_times)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add slowdown factor vs serial
    if i > 0:
        slowdown = time / training_times[0]
        ax3.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{slowdown:.2f}x slower',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
ax3.set_xlabel('Implementation', fontweight='bold')
ax3.set_title('Training Time Comparison\n(Most Computationally Intensive Phase)', 
              fontweight='bold', pad=20)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

plt.tight_layout()
plt.savefig('fig3_training_time.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig3_training_time.jpg")

# ============================================================================
# Figure 4: Speedup vs Serial Baseline
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

speedups = [total_times[0] / t for t in total_times]
colors_speedup = ['green' if s >= 1 else 'red' for s in speedups]

bars4 = ax4.barh(implementations, speedups, color=colors_speedup, 
                 edgecolor='black', linewidth=1.2, alpha=0.7)

# Add baseline line at 1.0
ax4.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Serial Baseline (1.0x)')

for i, (bar, speedup) in enumerate(zip(bars4, speedups)):
    width = bar.get_width()
    label_x = width + 0.05 if speedup < 1 else width - 0.05
    ha = 'left' if speedup < 1 else 'right'
    ax4.text(label_x, bar.get_y() + bar.get_height()/2.,
             f'{speedup:.2f}x',
             ha=ha, va='center', fontweight='bold', fontsize=10)

ax4.set_xlabel('Speedup Factor (higher is better)', fontweight='bold')
ax4.set_ylabel('Implementation', fontweight='bold')
ax4.set_title('Speedup vs Serial Baseline\n(Values < 1.0 indicate slowdown)', 
              fontweight='bold', pad=20)
ax4.legend(loc='lower right', framealpha=0.9)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.set_axisbelow(True)
ax4.set_xlim(0, max(speedups) * 1.15)

plt.tight_layout()
plt.savefig('fig4_speedup_comparison.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig4_speedup_comparison.jpg")

# ============================================================================
# Figure 5: Hybrid Implementations Only (Without Serial)
# ============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

hybrid_impl = implementations[1:]  # Exclude serial
hybrid_times = total_times[1:]
hybrid_colors = colors[1:]

bars5 = ax5.bar(hybrid_impl, hybrid_times, color=hybrid_colors, 
                edgecolor='black', linewidth=1.2)

for i, (bar, time) in enumerate(zip(bars5, hybrid_times)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight fastest hybrid
bars5[0].set_edgecolor('gold')
bars5[0].set_linewidth(3)

ax5.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax5.set_xlabel('Hybrid Implementation', fontweight='bold')
ax5.set_title('Hybrid Implementations Performance Comparison\n(Excluding Serial Baseline)', 
              fontweight='bold', pad=20)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.set_axisbelow(True)

# Add annotation for fastest
ax5.text(0.02, 0.98, '⭐ Fastest Hybrid Implementation', transform=ax5.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

plt.tight_layout()
plt.savefig('fig5_hybrid_only_comparison.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig5_hybrid_only_comparison.jpg")

# ============================================================================
# Figure 6: Performance Metrics Heatmap
# ============================================================================
fig6, ax6 = plt.subplots(figsize=(10, 7))

# Create data matrix (normalized to show relative performance)
data_matrix = np.array([
    total_times,
    training_times,
    load_times,
    eval_times
])

# Normalize each row to [0, 1] for better color visualization
data_normalized = np.zeros_like(data_matrix)
for i in range(data_matrix.shape[0]):
    row_min = data_matrix[i].min()
    row_max = data_matrix[i].max()
    if row_max > row_min:
        data_normalized[i] = (data_matrix[i] - row_min) / (row_max - row_min)
    else:
        data_normalized[i] = 0.5

# Create heatmap
im = ax6.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax6.set_xticks(np.arange(len(implementations)))
ax6.set_yticks(np.arange(len(['Total Time', 'Training Time', 'Load Time', 'Eval Time'])))
ax6.set_xticklabels(implementations)
ax6.set_yticklabels(['Total Time', 'Training Time', 'Load Time', 'Eval Time'])

# Rotate the tick labels for better readability
plt.setp(ax6.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations with actual values
for i in range(data_matrix.shape[0]):
    for j in range(data_matrix.shape[1]):
        text = ax6.text(j, i, f'{data_matrix[i, j]:.4f}s',
                       ha="center", va="center", color="black", fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax6.set_title('Performance Metrics Heatmap\n(Darker red = slower, Greener = faster)', 
              fontweight='bold', pad=20)
cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('Normalized Performance\n(0=best, 1=worst)', rotation=270, labelpad=20, fontweight='bold')

plt.tight_layout()
plt.savefig('fig6_performance_heatmap.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig6_performance_heatmap.jpg")

# ============================================================================
# Figure 7: Side-by-Side Comparison of Key Metrics
# ============================================================================
fig7, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total Time
axes[0, 0].bar(range(len(implementations)), total_times, color=colors, edgecolor='black')
axes[0, 0].set_title('Total Execution Time', fontweight='bold')
axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
axes[0, 0].set_xticks(range(len(implementations)))
axes[0, 0].set_xticklabels(implementations, rotation=45, ha='right')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(total_times):
    axes[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# Training Time
axes[0, 1].bar(range(len(implementations)), training_times, color=colors, edgecolor='black')
axes[0, 1].set_title('Training Time', fontweight='bold')
axes[0, 1].set_ylabel('Time (seconds)', fontweight='bold')
axes[0, 1].set_xticks(range(len(implementations)))
axes[0, 1].set_xticklabels(implementations, rotation=45, ha='right')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(training_times):
    axes[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# Load Time
axes[1, 0].bar(range(len(implementations)), load_times, color=colors, edgecolor='black')
axes[1, 0].set_title('Data Load Time', fontweight='bold')
axes[1, 0].set_ylabel('Time (seconds)', fontweight='bold')
axes[1, 0].set_xticks(range(len(implementations)))
axes[1, 0].set_xticklabels(implementations, rotation=45, ha='right')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(load_times):
    axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

# Speedup
speedups = [total_times[0] / t for t in total_times]
colors_speedup_chart = ['green' if s >= 1 else 'red' for s in speedups]
axes[1, 1].bar(range(len(implementations)), speedups, color=colors_speedup_chart, 
               edgecolor='black', alpha=0.7)
axes[1, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('Speedup vs Serial', fontweight='bold')
axes[1, 1].set_ylabel('Speedup Factor', fontweight='bold')
axes[1, 1].set_xticks(range(len(implementations)))
axes[1, 1].set_xticklabels(implementations, rotation=45, ha='right')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(speedups):
    axes[1, 1].text(i, v, f'{v:.2f}x', ha='center', va='bottom', fontsize=8)

fig7.suptitle('Comprehensive Performance Comparison\n(10,000 patients dataset)', 
              fontweight='bold', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('fig7_comprehensive_comparison.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig7_comprehensive_comparison.jpg")

# ============================================================================
# Figure 8: Thread Configuration vs Performance
# ============================================================================
fig8, ax8 = plt.subplots(figsize=(10, 6))

thread_configs = ['Serial\n(1 thread)', 'Pthread+MPI\n(8 threads)', 
                  'OpenMP+Pthread\n(4 threads)', 'Triple Hybrid\n(8 threads)', 
                  'OpenMP+MPI\n(4 threads)']
thread_counts = [1, 8, 4, 8, 4]

# Create scatter plot
scatter = ax8.scatter(thread_counts, total_times, c=total_times, cmap='RdYlGn_r', 
                      s=500, alpha=0.6, edgecolors='black', linewidth=2)

# Add labels for each point
for i, (tc, tt, label) in enumerate(zip(thread_counts, total_times, thread_configs)):
    ax8.annotate(label, (tc, tt), xytext=(10, 5), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax8.set_xlabel('Total Thread Count', fontweight='bold')
ax8.set_ylabel('Execution Time (seconds)', fontweight='bold')
ax8.set_title('Thread Count vs Performance\n(More threads ≠ Better performance)', 
              fontweight='bold', pad=20)
ax8.grid(True, alpha=0.3, linestyle='--')
ax8.set_xticks([1, 2, 4, 6, 8])

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('Execution Time (s)', rotation=270, labelpad=15, fontweight='bold')

plt.tight_layout()
plt.savefig('fig8_threads_vs_performance.jpg', bbox_inches='tight', quality=95)
plt.close()
print("✓ Generated: fig8_threads_vs_performance.jpg")

print("\n" + "="*60)
print("All graphs generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. fig1_total_execution_time.jpg - Overall performance comparison")
print("  2. fig2_execution_breakdown.jpg - Stacked time breakdown")
print("  3. fig3_training_time.jpg - Training phase comparison")
print("  4. fig4_speedup_comparison.jpg - Speedup/slowdown analysis")
print("  5. fig5_hybrid_only_comparison.jpg - Hybrid implementations only")
print("  6. fig6_performance_heatmap.jpg - Performance metrics heatmap")
print("  7. fig7_comprehensive_comparison.jpg - Multi-panel comparison")
print("  8. fig8_threads_vs_performance.jpg - Thread count analysis")
print("\nAll figures are publication-ready at 300 DPI in JPG format.")
print("="*60)
