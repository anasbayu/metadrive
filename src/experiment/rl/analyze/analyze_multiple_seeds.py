"""
Academic Visualization of PPO Training Runs from CSV Progress Files
Generates publication-quality plots matching TensorBoard style
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import glob

CSV_DIR = './file/logs/ppo_experiment'
OUTPUT_DIR = './file/logs/outputs/ppo_analysis'

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})


def load_ppo_runs(csv_dir=CSV_DIR):
    """
    Load all PPO training runs from CSV files
    
    Returns:
    --------
    runs : list of dict
        List containing {'steps': array, 'rewards': array, 'seed': str} for each run
    """
    csv_files = sorted(glob.glob(f'{csv_dir}/progress_seed*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No progress_seed*.csv files found in {csv_dir}")
    
    runs = []
    for csv_file in csv_files:
        # Extract seed number from filename
        seed = Path(csv_file).stem.replace('progress_seed', '')
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Extract timesteps and episode reward mean
        steps = df['time/total_timesteps'].values
        rewards = df['rollout/ep_rew_mean'].values
        
        runs.append({
            'steps': steps,
            'rewards': rewards,
            'seed': seed,
            'filename': Path(csv_file).name
        })
        
        print(f"  ✓ Loaded {Path(csv_file).name}: {len(steps)} data points, "
              f"{steps[-1]:,.0f} total steps")
    
    return runs


def smooth_curve(data, window_size=10):
    """Apply uniform smoothing to reduce noise in learning curves"""
    if len(data) < window_size:
        return data
    return uniform_filter1d(data, size=window_size, mode='nearest')


def plot_learning_curves(runs, output_path='ppo_learning_curves.pdf', 
                         smooth_window=10, show_individual=True, 
                         figsize=(10, 6)):
    """
    Create publication-quality plot of PPO learning curves matching TensorBoard style
    
    Parameters:
    -----------
    runs : list of dict
        List of runs from load_ppo_runs()
    output_path : str
        Path to save the figure
    smooth_window : int
        Window size for smoothing (default: 10)
    show_individual : bool
        Whether to show individual seed curves (default: True)
    figsize : tuple
        Figure size in inches (width, height)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette matching TensorBoard
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Find common step range for interpolation
    max_steps = max(run['steps'][-1] for run in runs)
    min_steps = min(run['steps'][0] for run in runs)
    
    # Create common step axis for computing mean/std
    common_steps = np.linspace(min_steps, max_steps, 1000)
    interpolated_rewards = []
    
    # Plot individual runs
    for i, run in enumerate(runs):
        steps = run['steps']
        rewards = run['rewards']
        
        # Smooth the rewards
        smoothed_rewards = smooth_curve(rewards, window_size=smooth_window)
        
        # Interpolate to common step axis for mean/std calculation
        interp_rewards = np.interp(common_steps, steps, smoothed_rewards)
        interpolated_rewards.append(interp_rewards)
        
        # Plot individual run
        if show_individual:
            ax.plot(steps, smoothed_rewards, color=colors[i % len(colors)], 
                   alpha=0.7, linewidth=1.8, 
                   label=f'Seed {run["seed"]}', zorder=3)
    
    # Calculate mean and std across runs
    interpolated_rewards = np.array(interpolated_rewards)
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_rewards = np.std(interpolated_rewards, axis=0)
    
    # Plot mean with confidence interval
    ax.plot(common_steps, mean_rewards, color='black', linewidth=2.5, 
           label='Mean', zorder=4, linestyle='-')
    ax.fill_between(common_steps, mean_rewards - std_rewards, 
                    mean_rewards + std_rewards, 
                    color='black', alpha=0.15, zorder=2, label='±1 Std Dev')
    
    # Formatting
    ax.set_xlabel('Training Steps', fontweight='normal')
    ax.set_ylabel('Episode Reward Mean', fontweight='normal')
    ax.set_title('PPO Training Performance Across Multiple Seeds', 
                fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax.minorticks_on()
    
    # Format x-axis to show in millions
    ax.ticklabel_format(style='plain', axis='x')
    
    def format_millions(x, pos):
        """Format axis labels in millions"""
        if x >= 1e6:
            return f'{int(x/1e6)}M'
        elif x >= 1e3:
            return f'{int(x/1e3)}K'
        else:
            return f'{int(x)}'
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(format_millions))
    
    # Legend
    if show_individual:
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', 
                 fancybox=False, shadow=False, ncol=2 if len(runs) > 3 else 1)
    else:
        ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', 
                 fancybox=False, shadow=False)
    
    # Set reasonable y-axis limits with some padding
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.05 * (y_max - y_min), 
                y_max + 0.05 * (y_max - y_min))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', 
                dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Figure saved to: {output_path}")
    print(f"✓ PNG version saved to: {output_path.replace('.pdf', '.png')}")
    
    return fig, ax


def plot_multi_panel(runs, output_path='ppo_multi_panel_analysis.pdf', 
                    smooth_window=10, figsize=(14, 10)):
    """
    Create multi-panel figure with comprehensive analysis
    
    Panels:
    1. Learning curves
    2. Final performance distribution
    3. Convergence analysis (running maximum)
    """
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Find common step range
    max_steps = max(run['steps'][-1] for run in runs)
    min_steps = min(run['steps'][0] for run in runs)
    common_steps = np.linspace(min_steps, max_steps, 1000)
    
    interpolated_rewards = []
    smoothed_runs = []
    
    for run in runs:
        smoothed = smooth_curve(run['rewards'], window_size=smooth_window)
        smoothed_runs.append({'steps': run['steps'], 'rewards': smoothed, 'seed': run['seed']})
        interp = np.interp(common_steps, run['steps'], smoothed)
        interpolated_rewards.append(interp)
    
    interpolated_rewards = np.array(interpolated_rewards)
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_rewards = np.std(interpolated_rewards, axis=0)
    
    # Panel 1: Learning Curves
    ax1 = fig.add_subplot(gs[0, :])
    
    for i, srun in enumerate(smoothed_runs):
        ax1.plot(srun['steps'], srun['rewards'], color=colors[i], 
                alpha=0.7, linewidth=1.5, label=f'Seed {srun["seed"]}')
    
    ax1.plot(common_steps, mean_rewards, 'k-', linewidth=2.5, 
            label='Mean', zorder=10)
    ax1.fill_between(common_steps, mean_rewards - std_rewards, 
                     mean_rewards + std_rewards,
                     color='black', alpha=0.15, label='±1 Std')
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('A) Learning Curves Across Seeds', fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', ncol=2)
    
    from matplotlib.ticker import FuncFormatter
    def format_millions(x, pos):
        if x >= 1e6:
            return f'{int(x/1e6)}M'
        elif x >= 1e3:
            return f'{int(x/1e3)}K'
        return f'{int(x)}'
    ax1.xaxis.set_major_formatter(FuncFormatter(format_millions))
    
    # Panel 2: Final Performance Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Get final 10% of data for each seed
    final_rewards = []
    for run in runs:
        final_10pct = int(len(run['rewards']) * 0.9)
        final_rewards.append(run['rewards'][final_10pct:])
    
    positions = range(len(runs))
    bp = ax2.boxplot(final_rewards, positions=positions, widths=0.6, 
                     patch_artist=True, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    final_means = [np.mean(fr) for fr in final_rewards]
    ax2.scatter(positions, final_means, color='red', s=80, 
               zorder=10, marker='D', label='Mean')
    ax2.axhline(np.mean(final_means), color='black', linestyle='--', 
               linewidth=2, alpha=0.5, label='Overall Mean')
    
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Final Episode Reward')
    ax2.set_title('B) Final Performance Distribution', fontweight='bold', loc='left')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f'S{run["seed"]}' for run in runs])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Panel 3: Convergence Analysis (Running Maximum)
    ax3 = fig.add_subplot(gs[1, 1])
    
    for i, srun in enumerate(smoothed_runs):
        running_max = np.maximum.accumulate(srun['rewards'])
        ax3.plot(srun['steps'], running_max, color=colors[i], 
                alpha=0.7, linewidth=1.5, label=f'Seed {srun["seed"]}')
    
    # Mean running max
    running_maxes = []
    for i in range(len(runs)):
        running_max = np.maximum.accumulate(interpolated_rewards[i])
        running_maxes.append(running_max)
    mean_running_max = np.mean(running_maxes, axis=0)
    
    ax3.plot(common_steps, mean_running_max, 'k-', linewidth=2.5, 
            label='Mean', zorder=10)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Best Reward Achieved')
    ax3.set_title('C) Convergence Analysis (Running Maximum)', 
                 fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', ncol=2)
    ax3.xaxis.set_major_formatter(FuncFormatter(format_millions))
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', 
                dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Multi-panel figure saved to: {output_path}")
    print(f"✓ PNG version saved to: {output_path.replace('.pdf', '.png')}")
    
    return fig


def print_statistics(runs):
    """Print summary statistics for the PPO runs"""
    
    print("\n" + "="*70)
    print("PPO TRAINING STATISTICS")
    print("="*70)
    
    print(f"\nNumber of seeds: {len(runs)}")
    
    for run in runs:
        print(f"  Seed {run['seed']}: {len(run['steps'])} data points, "
              f"{run['steps'][-1]:,.0f} total steps")
    
    # Final performance (last 10% of training)
    final_rewards = []
    for run in runs:
        final_10pct = int(len(run['rewards']) * 0.9)
        final_mean = np.mean(run['rewards'][final_10pct:])
        final_rewards.append(final_mean)
    
    final_rewards = np.array(final_rewards)
    
    print(f"\nFinal Performance (averaged over last 10% of training):")
    print(f"  Mean ± Std: {final_rewards.mean():.2f} ± {final_rewards.std():.2f}")
    print(f"  Min: {final_rewards.min():.2f}")
    print(f"  Max: {final_rewards.max():.2f}")
    print(f"  Median: {np.median(final_rewards):.2f}")
    
    # Best performance achieved
    best_rewards = np.array([run['rewards'].max() for run in runs])
    print(f"\nBest Performance Ever Achieved:")
    print(f"  Mean ± Std: {best_rewards.mean():.2f} ± {best_rewards.std():.2f}")
    print(f"  Max across all seeds: {best_rewards.max():.2f}")
    
    # Convergence analysis
    threshold = final_rewards.mean() * 0.9  # 90% of final performance
    convergence_steps = []
    for run in runs:
        steps_to_converge = np.where(run['rewards'] >= threshold)[0]
        if len(steps_to_converge) > 0:
            first_idx = steps_to_converge[0]
            convergence_steps.append(run['steps'][first_idx])
    
    if convergence_steps:
        print(f"\nConvergence to 90% of final performance:")
        print(f"  Mean steps: {np.mean(convergence_steps):,.0f}")
        print(f"  Std: {np.std(convergence_steps):,.0f}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main execution function"""
    
    # Load data
    print("Loading PPO training runs from CSV files...")
    print()
    runs = load_ppo_runs(CSV_DIR)
    
    # Print statistics
    print_statistics(runs)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Generate main learning curve plot
    print("Generating learning curves...")
    plot_learning_curves(
        runs, 
        output_path=str(output_dir / 'ppo_learning_curves.pdf'),
        smooth_window=10,
        show_individual=True,
        figsize=(10, 6)
    )
    
    # Generate multi-panel analysis
    print("\nGenerating multi-panel analysis...")
    plot_multi_panel(
        runs,
        output_path=str(output_dir / 'ppo_multi_panel_analysis.pdf'),
        smooth_window=10,
        figsize=(14, 10)
    )
    
    print("\n✓ All visualizations complete!")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()