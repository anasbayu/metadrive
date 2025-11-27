"""
Training Curves Visualization for Thesis
========================================
Author: Anas Bayu Kusuma
Date: 2025

This script generates publication-quality plots comparing PPO-Clip and Leaky PPO
from multiple training runs (5 seeds each).

Requirements:
- numpy
- pandas
- matplotlib
- seaborn
- scipy

Usage:
    python plot_training_curves.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths to your data
# Adjust these paths according to your actual file structure
DATA_DIR = Path("training_logs")  # Folder containing your CSV files
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Algorithm configurations
ALGORITHMS = {
    'PPO': {
        'color': 'steelblue',
        'label': 'PPO-Clip',
        'files': [
            'ppo_seed_42.csv',
            'ppo_seed_123.csv',
            'ppo_seed_456.csv',
            'ppo_seed_789.csv',
            'ppo_seed_1024.csv',
        ]
    },
    'Leaky': {
        'color': 'coral',
        'label': 'Leaky PPO',
        'files': [
            'leaky_ppo_seed_42.csv',
            'leaky_ppo_seed_123.csv',
            'leaky_ppo_seed_456.csv',
            'leaky_ppo_seed_789.csv',
            'leaky_ppo_seed_1024.csv',
        ]
    }
}

# Smoothing parameter (higher = smoother, but less detail)
SMOOTH_SIGMA = 2.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_training_logs(file_paths, data_dir):
    """
    Load training logs from CSV files
    
    Expected CSV format:
    - timestep: int (e.g., 50000, 100000, ...)
    - reward_mean: float (mean episode reward)
    - reward_std: float (standard deviation)
    - success_rate: float (0.0 to 1.0)
    - episode_length_mean: float (optional)
    
    Parameters:
    -----------
    file_paths : list of str
        List of CSV file names
    data_dir : Path
        Directory containing the CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with additional 'seed' column
    """
    dfs = []
    
    for i, file_path in enumerate(file_paths):
        full_path = data_dir / file_path
        
        if not full_path.exists():
            print(f"⚠️  Warning: File not found: {full_path}")
            continue
            
        df = pd.read_csv(full_path)
        df['seed'] = i  # Add seed identifier
        dfs.append(df)
        print(f"✅ Loaded: {file_path} ({len(df)} rows)")
    
    if not dfs:
        raise FileNotFoundError(f"No valid data files found in {data_dir}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def smooth_curve(data, sigma=2.0):
    """
    Apply Gaussian smoothing for better visualization
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of values to smooth
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns:
    --------
    np.ndarray
        Smoothed data
    """
    if len(data) < 3:  # Need at least 3 points to smooth
        return data
    return gaussian_filter1d(data, sigma=sigma, mode='nearest')


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval using percentile method
    
    Parameters:
    -----------
    data : np.ndarray
        Array of values (across seeds)
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
        
    Returns:
    --------
    tuple
        (mean, lower_bound, upper_bound)
    """
    mean = np.mean(data)
    
    # Compute SEM (Standard Error of Mean)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    
    # For 95% CI with t-distribution (more accurate for small n)
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    margin = t_value * sem
    lower = mean - margin
    upper = mean + margin
    
    return mean, lower, upper


def aggregate_runs(df, x_col='timestep', y_col='reward_mean'):
    """
    Aggregate data from multiple runs (seeds)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns [timestep, seed, reward_mean, ...]
    x_col : str
        Column name for x-axis (default: 'timestep')
    y_col : str
        Column name for y-axis values (default: 'reward_mean')
        
    Returns:
    --------
    tuple
        (timesteps, mean, lower_ci, upper_ci)
    """
    grouped = df.groupby(x_col)[y_col]
    
    timesteps = sorted(df[x_col].unique())
    means = []
    lowers = []
    uppers = []
    
    for timestep in timesteps:
        values = grouped.get_group(timestep).values
        mean, lower, upper = compute_confidence_interval(values)
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
    
    return np.array(timesteps), np.array(means), np.array(lowers), np.array(uppers)

# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plot_training_comparison(ppo_df, leaky_df, output_path='training_curves.png'):
    """
    Generate publication-quality training curves comparison
    
    Parameters:
    -----------
    ppo_df : pd.DataFrame
        PPO training data from all seeds
    leaky_df : pd.DataFrame
        Leaky PPO training data from all seeds
    output_path : str
        Path to save the output figure
    """
    
    # Create figure with 2 subplots (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # -------------------------------------------------------------------------
    # SUBPLOT 1: Episode Reward
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    print("\n📊 Processing reward curves...")
    
    # Aggregate data from multiple seeds
    ppo_timesteps, ppo_mean, ppo_lower, ppo_upper = aggregate_runs(
        ppo_df, x_col='timestep', y_col='reward_mean'
    )
    leaky_timesteps, leaky_mean, leaky_lower, leaky_upper = aggregate_runs(
        leaky_df, x_col='timestep', y_col='reward_mean'
    )
    
    # Smooth curves for better visualization
    ppo_smooth = smooth_curve(ppo_mean, sigma=SMOOTH_SIGMA)
    leaky_smooth = smooth_curve(leaky_mean, sigma=SMOOTH_SIGMA)
    
    # Also smooth CI bounds for consistency
    ppo_lower_smooth = smooth_curve(ppo_lower, sigma=SMOOTH_SIGMA)
    ppo_upper_smooth = smooth_curve(ppo_upper, sigma=SMOOTH_SIGMA)
    leaky_lower_smooth = smooth_curve(leaky_lower, sigma=SMOOTH_SIGMA)
    leaky_upper_smooth = smooth_curve(leaky_upper, sigma=SMOOTH_SIGMA)
    
    # Plot PPO
    ax1.plot(ppo_timesteps, ppo_smooth, 
             label='PPO-Clip', 
             linewidth=2.5, 
             color=ALGORITHMS['PPO']['color'], 
             alpha=0.9,
             zorder=3)
    ax1.fill_between(ppo_timesteps, ppo_lower_smooth, ppo_upper_smooth,
                      color=ALGORITHMS['PPO']['color'], 
                      alpha=0.2, 
                      label='95% CI (PPO)',
                      zorder=2)
    
    # Plot Leaky PPO
    ax1.plot(leaky_timesteps, leaky_smooth, 
             label='Leaky PPO', 
             linewidth=2.5, 
             color=ALGORITHMS['Leaky']['color'], 
             alpha=0.9,
             zorder=3)
    ax1.fill_between(leaky_timesteps, leaky_lower_smooth, leaky_upper_smooth,
                      color=ALGORITHMS['Leaky']['color'], 
                      alpha=0.2, 
                      label='95% CI (Leaky)',
                      zorder=2)
    
    # Add reference lines (Optuna best values) - optional
    if 'optuna_best' in ppo_df.columns:
        ppo_optuna_best = ppo_df['optuna_best'].iloc[0]
        ax1.axhline(y=ppo_optuna_best, color=ALGORITHMS['PPO']['color'], 
                   linestyle=':', alpha=0.4, linewidth=1.5)
    
    if 'optuna_best' in leaky_df.columns:
        leaky_optuna_best = leaky_df['optuna_best'].iloc[0]
        ax1.axhline(y=leaky_optuna_best, color=ALGORITHMS['Leaky']['color'], 
                   linestyle=':', alpha=0.4, linewidth=1.5)
    
    # Formatting
    ax1.set_ylabel('Mean Episode Reward', fontweight='bold')
    ax1.set_title('(a) Episode Reward vs Training Timesteps', pad=10)
    ax1.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set y-axis limits (adjust based on your data)
    reward_min = min(ppo_lower.min(), leaky_lower.min()) - 20
    reward_max = max(ppo_upper.max(), leaky_upper.max()) + 20
    ax1.set_ylim([reward_min, reward_max])
    
    print(f"  PPO final reward: {ppo_mean[-1]:.2f} ± {(ppo_upper[-1] - ppo_lower[-1])/2:.2f}")
    print(f"  Leaky PPO final reward: {leaky_mean[-1]:.2f} ± {(leaky_upper[-1] - leaky_lower[-1])/2:.2f}")
    
    # -------------------------------------------------------------------------
    # SUBPLOT 2: Success Rate
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    print("\n📊 Processing success rate curves...")
    
    # Aggregate success rate
    ppo_timesteps_sr, ppo_mean_sr, ppo_lower_sr, ppo_upper_sr = aggregate_runs(
        ppo_df, x_col='timestep', y_col='success_rate'
    )
    leaky_timesteps_sr, leaky_mean_sr, leaky_lower_sr, leaky_upper_sr = aggregate_runs(
        leaky_df, x_col='timestep', y_col='success_rate'
    )
    
    # Convert to percentage
    ppo_mean_sr *= 100
    ppo_lower_sr *= 100
    ppo_upper_sr *= 100
    leaky_mean_sr *= 100
    leaky_lower_sr *= 100
    leaky_upper_sr *= 100
    
    # Smooth curves
    ppo_smooth_sr = smooth_curve(ppo_mean_sr, sigma=SMOOTH_SIGMA)
    leaky_smooth_sr = smooth_curve(leaky_mean_sr, sigma=SMOOTH_SIGMA)
    ppo_lower_smooth_sr = smooth_curve(ppo_lower_sr, sigma=SMOOTH_SIGMA)
    ppo_upper_smooth_sr = smooth_curve(ppo_upper_sr, sigma=SMOOTH_SIGMA)
    leaky_lower_smooth_sr = smooth_curve(leaky_lower_sr, sigma=SMOOTH_SIGMA)
    leaky_upper_smooth_sr = smooth_curve(leaky_upper_sr, sigma=SMOOTH_SIGMA)
    
    # Plot PPO
    ax2.plot(ppo_timesteps_sr, ppo_smooth_sr, 
             label='PPO-Clip',
             linewidth=2.5, 
             color=ALGORITHMS['PPO']['color'], 
             alpha=0.9,
             zorder=3)
    ax2.fill_between(ppo_timesteps_sr, ppo_lower_smooth_sr, ppo_upper_smooth_sr,
                      color=ALGORITHMS['PPO']['color'], 
                      alpha=0.2,
                      zorder=2)
    
    # Plot Leaky PPO
    ax2.plot(leaky_timesteps_sr, leaky_smooth_sr, 
             label='Leaky PPO',
             linewidth=2.5, 
             color=ALGORITHMS['Leaky']['color'], 
             alpha=0.9,
             zorder=3)
    ax2.fill_between(leaky_timesteps_sr, leaky_lower_smooth_sr, leaky_upper_smooth_sr,
                      color=ALGORITHMS['Leaky']['color'], 
                      alpha=0.2,
                      zorder=2)
    
    # Formatting
    ax2.set_xlabel('Training Timesteps', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('(b) Success Rate vs Training Timesteps', pad=10)
    ax2.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set y-axis limits
    sr_min = max(0, min(ppo_lower_sr.min(), leaky_lower_sr.min()) - 5)
    sr_max = min(100, max(ppo_upper_sr.max(), leaky_upper_sr.max()) + 5)
    ax2.set_ylim([sr_min, sr_max])
    
    # Format x-axis with 'M' suffix for millions
    ax2.set_xticks(np.arange(0, 2.5e6, 0.5e6))
    ax2.set_xticklabels(['0', '0.5M', '1.0M', '1.5M', '2.0M'])
    ax2.set_xlim([0, 2e6])
    
    print(f"  PPO final success rate: {ppo_mean_sr[-1]:.1f}% ± {(ppo_upper_sr[-1] - ppo_lower_sr[-1])/2:.1f}%")
    print(f"  Leaky PPO final success rate: {leaky_mean_sr[-1]:.1f}% ± {(leaky_upper_sr[-1] - leaky_lower_sr[-1])/2:.1f}%")
    
    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    plt.tight_layout()
    
    # Save as PNG
    png_path = OUTPUT_DIR / output_path
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved PNG: {png_path}")
    
    # Save as PDF (vector format, better for LaTeX)
    pdf_path = OUTPUT_DIR / output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved PDF: {pdf_path}")
    
    # Save as SVG (editable in Inkscape)
    svg_path = OUTPUT_DIR / output_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✅ Saved SVG: {svg_path}")
    
    plt.close()
    
    return fig


# ============================================================================
# STATISTICS SUMMARY FUNCTION
# ============================================================================

def compute_summary_statistics(ppo_df, leaky_df):
    """
    Compute comprehensive summary statistics for thesis table
    
    Returns:
    --------
    dict
        Dictionary containing all metrics for both algorithms
    """
    
    print("\n" + "="*70)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*70)
    
    def get_final_metrics(df):
        """Extract final metrics from each seed"""
        seeds = df['seed'].unique()
        final_rewards = []
        final_success_rates = []
        
        for seed in seeds:
            seed_df = df[df['seed'] == seed]
            # Get last timestep for this seed
            final_rewards.append(seed_df['reward_mean'].iloc[-1])
            final_success_rates.append(seed_df['success_rate'].iloc[-1] * 100)
        
        return np.array(final_rewards), np.array(final_success_rates)
    
    # Get final metrics
    ppo_rewards, ppo_sr = get_final_metrics(ppo_df)
    leaky_rewards, leaky_sr = get_final_metrics(leaky_df)
    
    # Compute statistics
    stats = {
        'PPO': {
            'reward_mean': np.mean(ppo_rewards),
            'reward_std': np.std(ppo_rewards, ddof=1),
            'reward_min': np.min(ppo_rewards),
            'reward_max': np.max(ppo_rewards),
            'reward_cv': (np.std(ppo_rewards, ddof=1) / np.mean(ppo_rewards)) * 100,
            'success_rate_mean': np.mean(ppo_sr),
            'success_rate_std': np.std(ppo_sr, ddof=1),
        },
        'Leaky': {
            'reward_mean': np.mean(leaky_rewards),
            'reward_std': np.std(leaky_rewards, ddof=1),
            'reward_min': np.min(leaky_rewards),
            'reward_max': np.max(leaky_rewards),
            'reward_cv': (np.std(leaky_rewards, ddof=1) / np.mean(leaky_rewards)) * 100,
            'success_rate_mean': np.mean(leaky_sr),
            'success_rate_std': np.std(leaky_sr, ddof=1),
        }
    }
    
    # Compute differences
    stats['difference'] = {
        'reward_abs': stats['Leaky']['reward_mean'] - stats['PPO']['reward_mean'],
        'reward_pct': ((stats['Leaky']['reward_mean'] / stats['PPO']['reward_mean']) - 1) * 100,
        'success_rate_pp': stats['Leaky']['success_rate_mean'] - stats['PPO']['success_rate_mean'],
        'cv_improvement': ((stats['PPO']['reward_cv'] - stats['Leaky']['reward_cv']) / stats['PPO']['reward_cv']) * 100,
    }
    
    # Print summary table
    print("\n" + "-"*70)
    print("FINAL PERFORMANCE METRICS")
    print("-"*70)
    print(f"{'Metric':<30} {'PPO-Clip':<15} {'Leaky PPO':<15} {'Diff':<15}")
    print("-"*70)
    print(f"{'Mean Reward':<30} {stats['PPO']['reward_mean']:>8.2f} ± {stats['PPO']['reward_std']:<4.2f} "
          f"{stats['Leaky']['reward_mean']:>8.2f} ± {stats['Leaky']['reward_std']:<4.2f} "
          f"{stats['difference']['reward_abs']:>+7.2f}")
    print(f"{'Success Rate (%)':<30} {stats['PPO']['success_rate_mean']:>8.1f} ± {stats['PPO']['success_rate_std']:<4.1f} "
          f"{stats['Leaky']['success_rate_mean']:>8.1f} ± {stats['Leaky']['success_rate_std']:<4.1f} "
          f"{stats['difference']['success_rate_pp']:>+7.1f} pp")
    print(f"{'Min Reward (worst seed)':<30} {stats['PPO']['reward_min']:>14.2f} "
          f"{stats['Leaky']['reward_min']:>14.2f} "
          f"{stats['Leaky']['reward_min'] - stats['PPO']['reward_min']:>+7.2f}")
    print(f"{'Max Reward (best seed)':<30} {stats['PPO']['reward_max']:>14.2f} "
          f"{stats['Leaky']['reward_max']:>14.2f} "
          f"{stats['Leaky']['reward_max'] - stats['PPO']['reward_max']:>+7.2f}")
    print(f"{'CV (Coefficient of Var %)':<30} {stats['PPO']['reward_cv']:>14.2f} "
          f"{stats['Leaky']['reward_cv']:>14.2f} "
          f"{stats['difference']['cv_improvement']:>+6.1f}%")
    print("-"*70)
    print(f"Reward improvement: {stats['difference']['reward_pct']:+.2f}%")
    print(f"CV improvement: {stats['difference']['cv_improvement']:+.1f}%")
    print("="*70)
    
    # Save statistics to JSON
    stats_path = OUTPUT_DIR / 'summary_statistics.json'
    with open(stats_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_stats = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                json_stats[key] = {k: float(v) for k, v in value.items()}
            else:
                json_stats[key] = float(value) if isinstance(value, np.number) else value
        
        json.dump(json_stats, f, indent=4)
    print(f"\n✅ Saved statistics: {stats_path}")
    
    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to generate all visualizations"""
    
    print("\n" + "="*70)
    print("TRAINING CURVES VISUALIZATION SCRIPT")
    print("="*70)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n📂 Loading training logs...")
    
    try:
        ppo_df = load_training_logs(
            ALGORITHMS['PPO']['files'], 
            DATA_DIR
        )
        print(f"✅ PPO data loaded: {len(ppo_df)} total rows, {ppo_df['seed'].nunique()} seeds")
        
        leaky_df = load_training_logs(
            ALGORITHMS['Leaky']['files'], 
            DATA_DIR
        )
        print(f"✅ Leaky PPO data loaded: {len(leaky_df)} total rows, {leaky_df['seed'].nunique()} seeds")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n📝 Expected file structure:")
        print(f"   {DATA_DIR}/")
        for algo_name, algo_config in ALGORITHMS.items():
            for file_name in algo_config['files']:
                print(f"     ├── {file_name}")
        return
    
    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------
    print("\n🎨 Generating training curves...")
    plot_training_comparison(ppo_df, leaky_df, 'training_curves.png')
    
    # -------------------------------------------------------------------------
    # Compute statistics
    # -------------------------------------------------------------------------
    stats = compute_summary_statistics(ppo_df, leaky_df)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Output files saved in: {OUTPUT_DIR.absolute()}")
    print("   ├── training_curves.png")
    print("   ├── training_curves.pdf")
    print("   ├── training_curves.svg")
    print("   └── summary_statistics.json")
    print("\n💡 Tip: Use PDF for LaTeX, PNG for Word/PowerPoint")


if __name__ == "__main__":
    main()