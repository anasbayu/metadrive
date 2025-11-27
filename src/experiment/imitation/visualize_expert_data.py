import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
# Path to your NEW forced-2d dataset
DATA_PATH = "./file/expert_data/expert_metadrive_500k_1200eps_with_recovery_v3.npz" 
# =================================================

def visualize():
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found: {DATA_PATH}")
        return

    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)
    actions = data['actions']
    
    # === SHAPE FIX ===
    # If shape is (N, 1, 2), squeeze it to (N, 2)
    if actions.ndim == 3 and actions.shape[1] == 1:
        print(f"  Shape detected: {actions.shape} -> Squeezing to 2D...")
        actions = actions.squeeze(axis=1)
    
    print(f"  Final Action Shape: {actions.shape}")
    
    # Extract columns
    steering = actions[:, 0]
    acceleration = actions[:, 1]

    # --- PLOT SETUP ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. STEERING DISTRIBUTION
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(steering, bins=100, color='#3498db', log=True, alpha=0.8)
    ax1.set_title("Steering Distribution (Log Scale)")
    ax1.set_xlabel("Steering (Left <-> Right)")
    ax1.set_ylabel("Count (Log)")
    ax1.grid(True, alpha=0.3)
    # Highlight the "Straight" zone
    ax1.axvline(-0.05, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(0.05, color='r', linestyle='--', alpha=0.5)

    # 2. ACCELERATION DISTRIBUTION (The Moment of Truth)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(acceleration, bins=50, color='#2ecc71', alpha=0.8)
    ax2.set_title("Acceleration Distribution")
    ax2.set_xlabel("Throttle/Brake")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    
    if np.mean(acceleration) == 0.5 and np.std(acceleration) == 0:
        ax2.text(0.5, 0.5, "WARNING: CONSTANT 0.5 DETECTED", 
                 ha='center', color='red', weight='bold', transform=ax2.transAxes)

    # 3. SCATTER PLOT (Steering vs Accel)
    # We take a random sample of 5000 points to keep plot fast/clean
    idx = np.random.choice(len(steering), 5000, replace=False)
    ax3 = fig.add_subplot(gs[1, :])
    sc = ax3.scatter(steering[idx], acceleration[idx], alpha=0.2, c='purple', s=10)
    ax3.set_title("Steering vs. Acceleration Correlation (Sampled)")
    ax3.set_xlabel("Steering")
    ax3.set_ylabel("Acceleration")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_xlim(-1.1, 1.1)
    
    # Add quadrant labels
    ax3.text(0.8, 0.8, "Turn & Gas", ha='center', color='gray')
    ax3.text(0.8, -0.8, "Turn & Brake", ha='center', color='gray')
    ax3.text(0.0, 0.8, "Straight & Gas", ha='center', color='gray')

    plt.tight_layout()
    save_path = "data_visualization_report.png"
    plt.savefig(save_path)
    print(f"\n📊 Visualization saved to: {save_path}")
    print("Open this image to verify your data.")

    # --- TEXT REPORT ---
    straight_pct = np.sum(np.abs(steering) < 0.05) / len(steering) * 100
    print("\n=== QUICK REPORT ===")
    print(f"Straight Driving: {straight_pct:.1f}%")
    print(f"Acceleration Mean: {np.mean(acceleration):.4f}")
    print(f"Acceleration Std:  {np.std(acceleration):.4f}")
    if np.std(acceleration) < 0.01:
        print("⚠️ WARNING: Acceleration has very low variance. Agent might not learn to brake.")
    else:
        print("✅ Acceleration variance looks healthy.")

if __name__ == "__main__":
    visualize()