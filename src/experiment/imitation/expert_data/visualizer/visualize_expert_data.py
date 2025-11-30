import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
# Path to your clean expert dataset
DATA_PATH = "./file/expert_data/expert_metadrive_500k_1000scenarios_fixed.npz"
# =================================================

def visualize():
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found: {DATA_PATH}")
        print(f"   Please run collect_expert_data_fixed.py first")
        return

    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)
    
    # Load all data
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    
    print(f"\n{'='*60}")
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
    
    # === SHAPE FIX ===
    # If shape is (N, 1, 2), squeeze it to (N, 2)
    if actions.ndim == 3 and actions.shape[1] == 1:
        print(f"\n  ⚙️  Squeezing actions from {actions.shape} to 2D...")
        actions = actions.squeeze(axis=1)
    
    if observations.ndim == 3 and observations.shape[1] == 1:
        print(f"  ⚙️  Squeezing observations from {observations.shape} to 2D...")
        observations = observations.squeeze(axis=1)
    
    print(f"\nFinal shapes:")
    print(f"  Actions: {actions.shape}")
    print(f"  Observations: {observations.shape}")
    
    # Extract action dimensions
    steering = actions[:, 0]
    acceleration = actions[:, 1]
    
    # Calculate episode statistics
    episode_boundaries = np.where(dones.flatten())[0]
    num_episodes = len(episode_boundaries)
    
    episode_lengths = []
    episode_rewards = []
    
    start = 0
    for end in episode_boundaries:
        episode_lengths.append(end - start + 1)
        episode_rewards.append(rewards.flatten()[start:end+1].sum())
        start = end + 1
    
    print(f"\nEpisode statistics:")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Avg episode reward: {np.mean(episode_rewards):.2f}")
    print("="*60 + "\n")

    # --- PLOT SETUP ---
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. STEERING DISTRIBUTION
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(steering, bins=100, color='#3498db', alpha=0.8, edgecolor='black')
    ax1.set_title("Steering Distribution", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Steering (Left ← → Right)")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    # Highlight the "Straight" zone
    ax1.axvline(-0.05, color='r', linestyle='--', alpha=0.5, label='±0.05 "straight" zone')
    ax1.axvline(0.05, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Center')
    ax1.legend()

    # 2. ACCELERATION DISTRIBUTION
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(acceleration, bins=50, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.set_title("Acceleration Distribution", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Throttle/Brake")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax2.axvline(np.mean(acceleration), color='orange', linestyle='-', 
                linewidth=2, label=f'Mean: {np.mean(acceleration):.3f}')
    ax2.legend()
    
    # Check for constant acceleration
    if np.std(acceleration) < 0.01:
        ax2.text(0.5, 0.9, "⚠️ LOW VARIANCE", 
                 ha='center', va='top', color='red', weight='bold', 
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 3. REWARD DISTRIBUTION
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(rewards.flatten(), bins=50, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax3.set_title("Step Reward Distribution", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Reward")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)
    ax3.axvline(np.mean(rewards), color='orange', linestyle='-', 
                linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    ax3.legend()

    # 4. SCATTER PLOT (Steering vs Accel)
    # Sample for performance
    sample_size = min(10000, len(steering))
    idx = np.random.choice(len(steering), sample_size, replace=False)
    ax4 = fig.add_subplot(gs[1, :2])
    sc = ax4.scatter(steering[idx], acceleration[idx], alpha=0.3, c='purple', s=5)
    ax4.set_title("Steering vs. Acceleration Correlation", fontweight='bold', fontsize=12)
    ax4.set_xlabel("Steering")
    ax4.set_ylabel("Acceleration")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_xlim(-1.1, 1.1)
    
    # Add reference lines
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax4.text(0.7, 0.8, "Right Turn\n+ Throttle", ha='center', va='center', 
             color='darkblue', alpha=0.6, fontsize=9)
    ax4.text(-0.7, 0.8, "Left Turn\n+ Throttle", ha='center', va='center',
             color='darkblue', alpha=0.6, fontsize=9)
    ax4.text(0.0, 0.8, "Straight\n+ Throttle", ha='center', va='center',
             color='darkgreen', alpha=0.6, fontsize=9)
    ax4.text(0.0, -0.8, "Straight\n+ Brake", ha='center', va='center',
             color='darkred', alpha=0.6, fontsize=9)

    # 5. EPISODE LENGTHS
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(episode_lengths, bins=30, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax5.set_title("Episode Length Distribution", fontweight='bold', fontsize=12)
    ax5.set_xlabel("Episode Length (steps)")
    ax5.set_ylabel("Count")
    ax5.grid(True, alpha=0.3)
    ax5.axvline(np.mean(episode_lengths), color='orange', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax5.legend()

    # 6. EPISODE REWARDS
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(episode_rewards, bins=30, color='#f39c12', alpha=0.8, edgecolor='black')
    ax6.set_title("Episode Reward Distribution", fontweight='bold', fontsize=12)
    ax6.set_xlabel("Episode Total Reward")
    ax6.set_ylabel("Count")
    ax6.grid(True, alpha=0.3)
    ax6.axvline(np.mean(episode_rewards), color='red', linestyle='-',
                linewidth=2, label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax6.legend()

    # 7. ACTION TRAJECTORY (First 1000 steps)
    ax7 = fig.add_subplot(gs[2, 1:])
    plot_steps = min(1000, len(steering))
    steps = np.arange(plot_steps)
    ax7.plot(steps, steering[:plot_steps], alpha=0.7, label='Steering', linewidth=1)
    ax7.plot(steps, acceleration[:plot_steps], alpha=0.7, label='Acceleration', linewidth=1)
    ax7.set_title("Action Trajectory (First 1000 steps)", fontweight='bold', fontsize=12)
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Action Value")
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    ax7.set_ylim(-1.1, 1.1)

    # Add overall title
    fig.suptitle(f'Expert Data Visualization - {os.path.basename(DATA_PATH)}', 
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    save_path = "expert_data_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {save_path}\n")

    # --- DETAILED TEXT REPORT ---
    print("="*60)
    print("DETAILED ANALYSIS REPORT")
    print("="*60)
    
    # Steering analysis
    straight_pct = np.sum(np.abs(steering) < 0.05) / len(steering) * 100
    left_turn_pct = np.sum(steering < -0.1) / len(steering) * 100
    right_turn_pct = np.sum(steering > 0.1) / len(steering) * 100
    
    print(f"\n📐 STEERING:")
    print(f"  Straight driving (|s| < 0.05): {straight_pct:.1f}%")
    print(f"  Left turns (s < -0.1):          {left_turn_pct:.1f}%")
    print(f"  Right turns (s > 0.1):          {right_turn_pct:.1f}%")
    print(f"  Mean:                           {np.mean(steering):.4f}")
    print(f"  Std:                            {np.std(steering):.4f}")
    
    # Acceleration analysis
    throttle_pct = np.sum(acceleration > 0.1) / len(acceleration) * 100
    brake_pct = np.sum(acceleration < -0.1) / len(acceleration) * 100
    neutral_pct = np.sum(np.abs(acceleration) <= 0.1) / len(acceleration) * 100
    
    print(f"\n⚡ ACCELERATION:")
    print(f"  Throttle (a > 0.1):    {throttle_pct:.1f}%")
    print(f"  Neutral (|a| <= 0.1):  {neutral_pct:.1f}%")
    print(f"  Brake (a < -0.1):      {brake_pct:.1f}%")
    print(f"  Mean:                  {np.mean(acceleration):.4f}")
    print(f"  Std:                   {np.std(acceleration):.4f}")
    
    # Action variance check
    if np.std(acceleration) < 0.01:
        print(f"\n  ⚠️  WARNING: Very low acceleration variance!")
        print(f"      Agent might struggle to learn braking behavior.")
    else:
        print(f"\n  ✅ Acceleration variance looks healthy")
    
    # Reward analysis
    print(f"\n💰 REWARDS:")
    print(f"  Mean step reward:     {np.mean(rewards):.4f}")
    print(f"  Mean episode reward:  {np.mean(episode_rewards):.2f}")
    print(f"  Max episode reward:   {np.max(episode_rewards):.2f}")
    print(f"  Min episode reward:   {np.min(episode_rewards):.2f}")
    
    # Data quality check
    print(f"\n🔍 DATA QUALITY:")
    has_nan_obs = np.isnan(observations).any()
    has_nan_act = np.isnan(actions).any()
    has_inf_obs = np.isinf(observations).any()
    has_inf_act = np.isinf(actions).any()
    
    if has_nan_obs or has_nan_act or has_inf_obs or has_inf_act:
        print(f"  ❌ Contains NaN/Inf values!")
        print(f"     Obs: NaN={has_nan_obs}, Inf={has_inf_obs}")
        print(f"     Act: NaN={has_nan_act}, Inf={has_inf_act}")
    else:
        print(f"  ✅ No NaN or Inf values detected")
    
    # Overall assessment
    print(f"\n📊 OVERALL ASSESSMENT:")
    issues = []
    
    if np.mean(episode_rewards) < 200:
        issues.append("Low episode rewards - expert might not be performing well")
    if np.std(acceleration) < 0.01:
        issues.append("Very low acceleration variance - limited braking behavior")
    if straight_pct > 95:
        issues.append("Excessive straight driving - might lack scenario diversity")
    if has_nan_obs or has_nan_act or has_inf_obs or has_inf_act:
        issues.append("Data contains invalid values (NaN/Inf)")
    
    if len(issues) == 0:
        print(f"  ✅ Data looks excellent! Ready for BC training.")
    else:
        print(f"  ⚠️  Potential issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
    
    print("="*60 + "\n")
    
    print(f"✅ Visualization complete! Check {save_path}")
    print(f"\n📝 Next step: Use this data for BC training")
    print(f"   Make sure TRAFFIC_DENSITY in BC training script matches: 0.0")

if __name__ == "__main__":
    visualize()