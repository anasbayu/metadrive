import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIGURATION =================
DATA_PATH = "./file/expert_data/new_success_only/expert_metadrive_500k_5000eps_normalized.npz" 
# =================================================

def visualize_expert_data(path):
    print(f"Loading data from {path}...")
    try:
        data = np.load(path)

        actions = data['actions']
        print(f"Actions shape: {actions.shape}")
        # Squeeze out the n_envs dimension if present
        if actions.ndim == 3:  # Shape is (steps, n_envs, action_dim)
            actions = actions[:, 0, :]  # Take first env, shape becomes (steps, action_dim)
        elif actions.ndim == 2 and actions.shape[1] == 1:  # Shape is (steps, 1)
            # This means actions were flattened incorrectly
            raise ValueError(f"Actions have wrong shape: {actions.shape}. Expected (n, 2) but got (n, 1)")
        
        print(f"Actions shape after processing: {actions.shape}")

        if np.any(actions < -1.0) or np.any(actions > 1.0):
            print("⚠️  WARNING: Actions outside [-1, 1] range!")
            print(f"   Min: {actions.min()}, Max: {actions.max()}")

        rewards = data['rewards']
        if rewards.ndim == 2:
            rewards = rewards.flatten()  # Convert (n, 1) to (n,)
        
        dones = data['dones']
    except FileNotFoundError:
        print("❌ File not found! Please check the path.")
        return

    print(f"Loaded {len(actions)} transitions.")

    # Extract Individual Actions
    # MetaDrive Action Space: [Steering (-1 to 1), Throttle/Brake (-1 to 1)]
    steering = actions[:, 0]
    throttle_brake = actions[:, 1]

    # Calculate Episode Metrics
    # Find indices where episodes end (done=True)
    done_indices = np.where(dones)[0]
    
    # Calculate lengths and returns
    episode_lengths = []
    episode_returns = []
    current_idx = 0
    
    for end_idx in done_indices:
        # Length is difference between indices
        length = end_idx - current_idx + 1
        episode_lengths.append(length)
        
        # Sum rewards for this slice
        ep_return = np.sum(rewards[current_idx : end_idx+1])
        episode_returns.append(ep_return)
        
        current_idx = end_idx + 1

    # ================= STATISTICS =================
    observations = data['observations']

    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Observations - Mean: {observations.mean():.4f} (should be ~0)")
    print(f"Observations - Std:  {observations.std():.4f} (should be ~1)")
    print(f"Actions - Mean:      {actions.mean(axis=0)} (steering, throttle)")
    print(f"Rewards - Mean:      {rewards.mean():.4f}")
    print(f"Episodes:            {len(episode_lengths)}")
    print(f"Avg Episode Length:  {np.mean(episode_lengths):.1f}")
    print(f"Avg Episode Return:  {np.mean(episode_returns):.1f}")
    print("="*60 + "\n")
    # ================ END OF STATISTICS =================

    print(f"\n📊 Episode Analysis:")
    print(f"  Total Episodes: {len(episode_lengths)}")
    print(f"  Shortest: {np.min(episode_lengths)} steps")
    print(f"  Longest:  {np.max(episode_lengths)} steps")
    print(f"  Mean:     {np.mean(episode_lengths):.1f} steps")
    print(f"  Best Return:  {np.max(episode_returns):.2f}")
    print(f"  Worst Return: {np.min(episode_returns):.2f}")


    # ================= PLOTTING =================
    plt.figure(figsize=(18, 10))
    plt.suptitle(f"Expert Data Analysis: {len(actions)} Steps, {len(episode_lengths)} Episodes", fontsize=16)

    # Plot 1: Steering Distribution
    plt.subplot(2, 3, 1)
    sns.histplot(steering, bins=50, kde=True, color="blue")
    plt.title("Steering Distribution\n(Should be centered at 0)")
    plt.xlabel("Steering Angle [-1, 1]")
    plt.ylabel("Count")

    # Plot 2: Throttle/Brake Distribution
    plt.subplot(2, 3, 2)
    sns.histplot(throttle_brake, bins=50, kde=True, color="green")
    plt.title("Throttle/Brake Distribution\n(>0 Accel, <0 Brake)")
    plt.xlabel("Throttle/Brake [-1, 1]")

    # Plot 3: Joint Action Distribution (The "Driving Envelope")
    plt.subplot(2, 3, 3)
    plt.hexbin(steering, throttle_brake, gridsize=30, cmap='inferno', mincnt=1)
    plt.colorbar(label='Count')
    plt.title("Joint Distribution: Steering vs Throttle\n(Look for 'V' shape)")
    plt.xlabel("Steering")
    plt.ylabel("Throttle")

    # Plot 4: Episode Lengths
    plt.subplot(2, 3, 4)
    sns.histplot(episode_lengths, bins=30, color="purple")
    plt.title("Episode Lengths\n(Short bars = Crashes?)")
    plt.xlabel("Steps per Episode")

    # Plot 5: Episode Returns (Total Reward)
    plt.subplot(2, 3, 5)
    sns.histplot(episode_returns, bins=30, color="orange")
    plt.title("Total Reward per Episode")
    plt.xlabel("Cumulative Reward")

    # Plot 6: Sample Trajectory (Actions over Time)
    plt.subplot(2, 3, 6)
    sample_length = min(500, len(steering))  # First 500 steps
    plt.plot(steering[:sample_length], label='Steering', alpha=0.7)
    plt.plot(throttle_brake[:sample_length], label='Throttle', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title("Sample Trajectory (First 500 steps)")
    plt.xlabel("Time Step")
    plt.ylabel("Action Value")
    plt.legend()
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # ================ END OF PLOTTING =================

    # ================= SAVING =================
    # Create a filename based on the input .npz name
    base_dir = os.path.dirname(path)
    base_name = os.path.basename(path).replace('.npz', '')
    save_path = os.path.join(base_dir, f"{base_name}_analysis.png")

    print(f"Saving graph to: {save_path}")
    
    # bbox_inches='tight' ensures labels don't get cut off
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("✅ Graph saved successfully!")
    plt.show()

if __name__ == "__main__":
    visualize_expert_data(DATA_PATH)