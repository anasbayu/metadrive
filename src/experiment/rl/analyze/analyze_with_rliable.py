"""
RLiable Analysis: PPO vs LeakyPPO Comparison
=============================================

This script analyzes experimental results comparing PPO and LeakyPPO,
each with 5 runs from different seeds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

SAVE_PATH = './file/rliable/images/'

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# STEP 1: Load Your Data
# ============================================================================
print("="*80)
print("LOADING DATA FROM JSON")
print("="*80)

with open('./file/rliable/optuna_experiment_results.json', 'r') as f:
    data = json.load(f)

print(f"Algorithms found: {list(data.keys())}")
print(f"Number of seeds per algorithm: {len(data['PPO'])}")
print(f"Episodes per seed: {len(data['PPO'][0])}")

# ============================================================================
# STEP 2: Convert to RLiable Format
# ============================================================================
# RLiable expects: (num_seeds, num_tasks_or_timesteps)
# Your data is: list of lists, where each inner list is one seed's results

score_dict = {
    'PPO': np.array(data['PPO']),
    'LeakyPPO': np.array(data['LeakyPPO'])
}

print(f"\nPPO shape: {score_dict['PPO'].shape}")
print(f"LeakyPPO shape: {score_dict['LeakyPPO'].shape}")

# Quick statistics
print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)
for algo_name in ['PPO', 'LeakyPPO']:
    scores = score_dict[algo_name]
    print(f"\n{algo_name}:")
    print(f"  Overall mean ± std:  {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"  Min: {scores.min():.2f}, Max: {scores.max():.2f}")
    print(f"  Per-seed means: {scores.mean(axis=1)}")

# ============================================================================
# STEP 3: Aggregate Metrics with Bootstrap Confidence Intervals
# ============================================================================
print("\n" + "="*80)
print("ROBUST AGGREGATE METRICS (with Bootstrap CIs)")
print("="*80)

# Define aggregate functions
def compute_metrics(scores):
    """Compute multiple aggregate metrics."""
    return np.array([
        metrics.aggregate_mean(scores),
        metrics.aggregate_median(scores),
        metrics.aggregate_iqm(scores),  # Interquartile Mean (recommended)
        metrics.aggregate_optimality_gap(scores)
    ])

# Compute with bootstrap confidence intervals
aggregate_scores, aggregate_cis = rly.get_interval_estimates(
    score_dict,
    compute_metrics,
    reps=2000  # Number of bootstrap replications
)

# Display results
metric_names = ['Mean', 'Median', 'IQM', 'Optimality Gap']
print("\nPPO:")
for i, name in enumerate(metric_names):
    value = aggregate_scores['PPO'][i]
    ci_low, ci_high = aggregate_cis['PPO'][:, i]
    print(f"  {name:20s}: {value:8.2f}  [95% CI: {ci_low:8.2f}, {ci_high:8.2f}]")

print("\nLeakyPPO:")
for i, name in enumerate(metric_names):
    value = aggregate_scores['LeakyPPO'][i]
    ci_low, ci_high = aggregate_cis['LeakyPPO'][:, i]
    print(f"  {name:20s}: {value:8.2f}  [95% CI: {ci_low:8.2f}, {ci_high:8.2f}]")

# ============================================================================
# STEP 4: Statistical Comparison
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL COMPARISON")
print("="*80)

# Compare IQM values
ppo_iqm = aggregate_scores['PPO'][2]
leaky_iqm = aggregate_scores['LeakyPPO'][2]
ppo_iqm_ci = aggregate_cis['PPO'][:, 2]
leaky_iqm_ci = aggregate_cis['LeakyPPO'][:, 2]

print(f"\nIQM Comparison:")
print(f"  PPO:       {ppo_iqm:8.2f}  [{ppo_iqm_ci[0]:8.2f}, {ppo_iqm_ci[1]:8.2f}]")
print(f"  LeakyPPO:  {leaky_iqm:8.2f}  [{leaky_iqm_ci[0]:8.2f}, {leaky_iqm_ci[1]:8.2f}]")
print(f"  Difference: {leaky_iqm - ppo_iqm:8.2f}")

# Check if confidence intervals overlap
ci_overlap = not (ppo_iqm_ci[1] < leaky_iqm_ci[0] or leaky_iqm_ci[1] < ppo_iqm_ci[0])
if ci_overlap:
    print(f"\n  ⚠ Confidence intervals OVERLAP - difference may not be statistically significant")
else:
    print(f"\n  ✓ Confidence intervals DO NOT overlap - difference is likely significant")
    if leaky_iqm > ppo_iqm:
        print(f"  → LeakyPPO appears to outperform PPO")
    else:
        print(f"  → PPO appears to outperform LeakyPPO")

# Bootstrap-based probability of improvement
print("\n" + "="*80)
print("PROBABILITY OF IMPROVEMENT")
print("="*80)

# Compute using overall scores
n_bootstrap = 5000
improvements = []
for _ in range(n_bootstrap):
    ppo_sample = np.random.choice(score_dict['PPO'].flatten(), 
                                   size=score_dict['PPO'].size, 
                                   replace=True)
    leaky_sample = np.random.choice(score_dict['LeakyPPO'].flatten(), 
                                     size=score_dict['LeakyPPO'].size, 
                                     replace=True)
    improvements.append(metrics.aggregate_iqm(leaky_sample.reshape(1, -1)) > 
                       metrics.aggregate_iqm(ppo_sample.reshape(1, -1)))

prob_improvement = np.mean(improvements)
print(f"P(LeakyPPO IQM > PPO IQM): {prob_improvement:.3f}")

if prob_improvement > 0.95:
    print("→ Very strong evidence that LeakyPPO is better")
elif prob_improvement > 0.75:
    print("→ Strong evidence that LeakyPPO is better")
elif prob_improvement > 0.5:
    print("→ Weak evidence that LeakyPPO is better")
elif prob_improvement > 0.25:
    print("→ Weak evidence that PPO is better")
else:
    print("→ Strong evidence that PPO is better")

# ============================================================================
# STEP 5: Performance Profiles
# ============================================================================
print("\n" + "="*80)
print("COMPUTING PERFORMANCE PROFILES...")
print("="*80)

# Use final performance (last 10% of episodes)
final_window = int(0.1 * score_dict['PPO'].shape[1])
final_scores = {
    'PPO': score_dict['PPO'][:, -final_window:].mean(axis=1).reshape(-1, 1),
    'LeakyPPO': score_dict['LeakyPPO'][:, -final_window:].mean(axis=1).reshape(-1, 1)
}

# Create thresholds
all_scores = np.concatenate([final_scores['PPO'], final_scores['LeakyPPO']])
score_thresholds = np.linspace(all_scores.min() - 10, all_scores.max() + 10, 100)

# Compute performance profiles
perf_profiles, perf_profile_cis = rly.create_performance_profile(
    final_scores,
    score_thresholds,
    reps=2000
)

# Plot performance profiles
fig, ax = plt.subplots(figsize=(10, 6))
plot_utils.plot_performance_profiles(
    perf_profiles,
    score_thresholds,
    performance_profile_cis=perf_profile_cis,
    colors={'PPO': 'C0', 'LeakyPPO': 'C1'},
    xlabel='Score Threshold (Final Performance)',
    ax=ax
)
ax.set_title('Performance Profiles: PPO vs LeakyPPO', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}performance_profiles_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {SAVE_PATH}performance_profiles_comparison.png")

# ============================================================================
# STEP 6: Sample Efficiency Curves
# ============================================================================
print("\n" + "="*80)
print("COMPUTING SAMPLE EFFICIENCY CURVES...")
print("="*80)

# Downsample for cleaner plotting
downsample_rate = 5  # Use every 5th episode
n_points = score_dict['PPO'].shape[1] // downsample_rate
downsampled_scores = {
    'PPO': score_dict['PPO'][:, ::downsample_rate],
    'LeakyPPO': score_dict['LeakyPPO'][:, ::downsample_rate]
}

# Compute IQM over time
def iqm_over_time(scores):
    return np.array([
        metrics.aggregate_iqm(scores[:, i:i+1]) 
        for i in range(scores.shape[1])
    ])

efficiency_scores, efficiency_cis = rly.get_interval_estimates(
    downsampled_scores,
    iqm_over_time,
    reps=1000
)

# Plot sample efficiency
episodes = np.arange(0, score_dict['PPO'].shape[1], downsample_rate)
fig, ax = plt.subplots(figsize=(12, 6))

for algo, color in [('PPO', 'C0'), ('LeakyPPO', 'C1')]:
    ax.plot(episodes, efficiency_scores[algo], label=f'{algo} (IQM)', 
            linewidth=2, color=color)
    ax.fill_between(
        episodes,
        efficiency_cis[algo][0],
        efficiency_cis[algo][1],
        alpha=0.25,
        color=color
    )

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Return (IQM)', fontsize=12)
ax.set_title('Sample Efficiency: PPO vs LeakyPPO', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}sample_efficiency_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {SAVE_PATH}sample_efficiency_comparison.png")

# ============================================================================
# STEP 7: Comprehensive Summary Visualization
# ============================================================================
print("\n" + "="*80)
print("CREATING COMPREHENSIVE SUMMARY...")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Learning curves (all seeds)
ax1 = fig.add_subplot(gs[0, 0])
for i in range(5):
    ax1.plot(score_dict['PPO'][i], alpha=0.4, color='C0', linewidth=1)
    if i == 0:
        ax1.plot([], [], alpha=0.4, color='C0', linewidth=2, label='PPO (individual seeds)')
ax1.plot(score_dict['PPO'].mean(axis=0), color='C0', linewidth=2.5, label='PPO (mean)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Return')
ax1.set_title('PPO: Learning Curves (5 Seeds)', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
for i in range(5):
    ax2.plot(score_dict['LeakyPPO'][i], alpha=0.4, color='C1', linewidth=1)
    if i == 0:
        ax2.plot([], [], alpha=0.4, color='C1', linewidth=2, label='LeakyPPO (individual seeds)')
ax2.plot(score_dict['LeakyPPO'].mean(axis=0), color='C1', linewidth=2.5, label='LeakyPPO (mean)')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Return')
ax2.set_title('LeakyPPO: Learning Curves (5 Seeds)', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 2: Aggregate metrics comparison with error bars
ax3 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(metric_names))
width = 0.35

for i, (algo, color) in enumerate([('PPO', 'C0'), ('LeakyPPO', 'C1')]):
    means = aggregate_scores[algo]
    ci_low = aggregate_cis[algo][0]
    ci_high = aggregate_cis[algo][1]
    errors = np.array([means - ci_low, ci_high - means])
    
    offset = width/2 if i == 0 else -width/2
    ax3.bar(x_pos + offset, means, width, alpha=0.7, color=color, 
            edgecolor='black', label=algo)
    ax3.errorbar(x_pos + offset, means, yerr=errors, fmt='none', 
                 ecolor='black', capsize=4, linewidth=1.5)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(metric_names, rotation=0)
ax3.set_ylabel('Score')
ax3.set_title('Aggregate Metrics with 95% Bootstrap CIs', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 3: Distribution of final scores (violin plot)
ax4 = fig.add_subplot(gs[1, 1])
final_ppo = final_scores['PPO'].flatten()
final_leaky = final_scores['LeakyPPO'].flatten()

positions = [1, 2]
data_to_plot = [final_ppo, final_leaky]
colors_violin = ['C0', 'C1']

parts = ax4.violinplot(data_to_plot, positions=positions, widths=0.6,
                       showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_violin[i])
    pc.set_alpha(0.6)

ax4.set_xticks(positions)
ax4.set_xticklabels(['PPO', 'LeakyPPO'])
ax4.set_ylabel('Final Return (Last 10% Episodes)')
ax4.set_title('Distribution of Final Performance', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 4: Sample efficiency comparison (zoomed)
ax5 = fig.add_subplot(gs[2, :])
for algo, color in [('PPO', 'C0'), ('LeakyPPO', 'C1')]:
    ax5.plot(episodes, efficiency_scores[algo], label=algo, 
            linewidth=2.5, color=color)
    ax5.fill_between(
        episodes,
        efficiency_cis[algo][0],
        efficiency_cis[algo][1],
        alpha=0.2,
        color=color
    )

ax5.set_xlabel('Episode', fontsize=12)
ax5.set_ylabel('Return (IQM)', fontsize=12)
ax5.set_title('Sample Efficiency Comparison (IQM with 95% CIs)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

plt.savefig(f'{SAVE_PATH}comprehensive_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {SAVE_PATH}comprehensive_comparison.png")

# ============================================================================
# STEP 8: Generate Summary Report
# ============================================================================
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

print(f"""
EXPERIMENT: PPO vs LeakyPPO Comparison
Number of seeds: 5 per algorithm
Episodes per seed: {score_dict['PPO'].shape[1]}

RESULTS (IQM - Interquartile Mean):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PPO:       {ppo_iqm:8.2f}  [95% CI: {ppo_iqm_ci[0]:8.2f}, {ppo_iqm_ci[1]:8.2f}]
  LeakyPPO:  {leaky_iqm:8.2f}  [95% CI: {leaky_iqm_ci[0]:8.2f}, {leaky_iqm_ci[1]:8.2f}]
  
  Difference: {leaky_iqm - ppo_iqm:8.2f} ({((leaky_iqm - ppo_iqm)/abs(ppo_iqm)*100):+.2f}%)

STATISTICAL SIGNIFICANCE:
  P(LeakyPPO > PPO): {prob_improvement:.3f}
  CI Overlap: {'Yes' if ci_overlap else 'No'}

CONCLUSION:
""")

if prob_improvement > 0.95 and not ci_overlap:
    print("  ✓ STRONG evidence that LeakyPPO outperforms PPO")
    print("    - Very high probability of improvement (>95%)")
    print("    - Non-overlapping confidence intervals")
elif prob_improvement > 0.75:
    print("  ⚠ MODERATE evidence that LeakyPPO outperforms PPO")
    print("    - High probability of improvement (75-95%)")
    if ci_overlap:
        print("    - But confidence intervals overlap - more data recommended")
elif prob_improvement > 0.5:
    print("  ⚠ WEAK evidence favoring LeakyPPO")
    print("    - Small probability of improvement (50-75%)")
    print("    - Results are not conclusive - consider more seeds")
elif prob_improvement > 0.25:
    print("  ⚠ WEAK evidence favoring PPO")
    print("    - Results are not conclusive - consider more seeds")
else:
    print("  ✓ Evidence suggests PPO may be better")
    print("    - LeakyPPO shows lower performance")

print(f"""
REPORTING FOR PAPER/PRESENTATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use this format:

  "We compared PPO and LeakyPPO across 5 random seeds. 
   PPO achieved an IQM of {ppo_iqm:.1f} (95% CI: [{ppo_iqm_ci[0]:.1f}, {ppo_iqm_ci[1]:.1f}]), 
   while LeakyPPO achieved {leaky_iqm:.1f} (95% CI: [{leaky_iqm_ci[0]:.1f}, {leaky_iqm_ci[1]:.1f}]).
   {
   'LeakyPPO demonstrated statistically significant improvement over PPO.' 
   if prob_improvement > 0.95 and not ci_overlap 
   else 'The difference was not statistically significant at the 95% confidence level.'
   }"

FILES GENERATED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. performance_profiles_comparison.png - Shows threshold probabilities
  2. sample_efficiency_comparison.png     - Learning curves with CIs
  3. comprehensive_comparison.png         - Full summary visualization

All saved to: {SAVE_PATH}
""")

print("\n✓ Analysis complete!")