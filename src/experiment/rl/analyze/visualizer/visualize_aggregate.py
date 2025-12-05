import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

# ============= CONFIGURATION =============
BASE_DIR = "./logs"  # Base directory containing algorithm folders

ALGORITHMS = {
    "PPO": {
        "path": "PPO",
        "color": "#2E86AB",
        "label": "PPO"
    },
    "LeakyPPO": {
        "path": "LeakyPPO", 
        "color": "#A23B72",
        "label": "LeakyPPO"
    },
    "PPO_Warmstart": {
        "path": "PPO_Warmstart",
        "color": "#F18F01",
        "label": "PPO + BC Warmstart"
    },
    "LeakyPPO_Warmstart": {
        "path": "LeakyPPO_Warmstart",
        "color": "#C73E1D",
        "label": "LeakyPPO + BC Warmstart"
    }
}

RUNS_PER_ALGO = 5  # Number of seeds/runs per algorithm
REWARD_COLUMN = "rollout/ep_rew_mean"
TIMESTEP_COLUMN = "time/total_timesteps"

OUTPUT_FILE = "learning_curves_comparison.png"
# =========================================

def load_run_data(algo_path, run_id):
    """Load progress.csv for a specific run"""
    csv_path = Path(BASE_DIR) / algo_path / f"run_{run_id}" / "progress.csv"
    
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df[[TIMESTEP_COLUMN, REWARD_COLUMN]].dropna()
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return None

def interpolate_to_common_timesteps(dfs, common_timesteps):
    """
    Interpolate all runs to common timestep grid for averaging
    """
    interpolated_rewards = []
    
    for df in dfs:
        if df is None or len(df) < 2:
            continue
            
        timesteps = df[TIMESTEP_COLUMN].values
        rewards = df[REWARD_COLUMN].values
        
        # Create interpolation function
        f = interp1d(timesteps, rewards, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
        
        # Interpolate to common grid
        interp_rewards = f(common_timesteps)
        interpolated_rewards.append(interp_rewards)
    
    return np.array(interpolated_rewards)

def load_algorithm_data(algo_config):
    """Load all runs for an algorithm and compute mean + std"""
    algo_path = algo_config["path"]
    
    print(f"\n📂 Loading {algo_config['label']}...")
    
    # Load all runs
    dfs = []
    for run_id in range(RUNS_PER_ALGO):
        df = load_run_data(algo_path, run_id)
        if df is not None:
            dfs.append(df)
            print(f"  ✓ Run {run_id}: {len(df)} datapoints")
    
    if len(dfs) == 0:
        print(f"  ❌ No valid runs found!")
        return None
    
    # Find common timestep range (intersection of all runs)
    min_timestep = max([df[TIMESTEP_COLUMN].min() for df in dfs])
    max_timestep = min([df[TIMESTEP_COLUMN].max() for df in dfs])
    
    # Create common timestep grid
    common_timesteps = np.linspace(min_timestep, max_timestep, 500)
    
    # Interpolate all runs to common grid
    rewards_matrix = interpolate_to_common_timesteps(dfs, common_timesteps)
    
    if len(rewards_matrix) == 0:
        print(f"  ❌ Interpolation failed!")
        return None
    
    # Compute statistics
    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    
    print(f"  ✓ Computed mean/std from {len(dfs)} runs")
    print(f"  ✓ Timestep range: {min_timestep:,} - {max_timestep:,}")
    
    return {
        "timesteps": common_timesteps,
        "mean": mean_rewards,
        "std": std_rewards,
        "n_runs": len(dfs)
    }

def plot_learning_curves(algo_data_dict):
    """Create publication-quality learning curves plot"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo_name, algo_config in ALGORITHMS.items():
        data = algo_data_dict.get(algo_name)
        
        if data is None:
            continue
        
        timesteps = data["timesteps"]
        mean = data["mean"]
        std = data["std"]
        color = algo_config["color"]
        label = f"{algo_config['label']} (n={data['n_runs']})"
        
        # Plot mean line
        ax.plot(timesteps, mean, color=color, linewidth=2.5, label=label, alpha=0.9)
        
        # Plot std area
        ax.fill_between(
            timesteps,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            linewidth=0
        )
    
    # Styling
    ax.set_xlabel("Training Timesteps", fontsize=14, fontweight='bold')
    ax.set_ylabel("Episode Reward (Mean)", fontsize=14, fontweight='bold')
    ax.set_title("Learning Curves: Algorithm Comparison", 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format x-axis (millions)
    ax.ticklabel_format(style='plain', axis='x')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))
    
    plt.tight_layout()
    
    return fig

def main():
    """Main execution"""
    
    print("="*60)
    print("LEARNING CURVES PLOTTER")
    print("="*60)
    
    # Load data for all algorithms
    algo_data = {}
    
    for algo_name, algo_config in ALGORITHMS.items():
        data = load_algorithm_data(algo_config)
        if data is not None:
            algo_data[algo_name] = data
    
    if len(algo_data) == 0:
        print("\n❌ No data loaded! Check your paths and file structure.")
        return
    
    print("\n" + "="*60)
    print(f"✅ Successfully loaded {len(algo_data)} algorithms")
    print("="*60)
    
    # Plot
    print(f"\n📊 Creating plot...")
    fig = plot_learning_curves(algo_data)
    
    # Save
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {OUTPUT_FILE}")
    
    plt.show()

if __name__ == "__main__":
    main()