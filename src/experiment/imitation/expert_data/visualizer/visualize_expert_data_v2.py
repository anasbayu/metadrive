import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIGURATION =================
DATA_PATH = "./file/expert_data/expert_metadrive_500k_noisy_v5.npz" 
# =================================================

def visualize_expert_data(path):
    print(f"Loading data from {path}...")
    try:
        data = np.load(path)
        actions = data['actions']
        rewards = data['rewards']
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

    plt.tight_layout()

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