import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from metadrive.envs.metadrive_env import MetaDriveEnv
import os

# ================= CONFIGURATION =================
MODEL_PATH = "./file/bc/models/bc_agent_forced2d.pth"
STATS_PATH = "./file/bc/models/normalization_stats.npz"
SAVE_PLOT_PATH = "agent_telemetry.png"
# =================================================

# Define Agent (Must match training)
class BCAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(BCAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh() 
        )
    def forward(self, x):
        return self.net(x)

def plot_telemetry():
    # 1. Setup
    if not os.path.exists(STATS_PATH):
        print("❌ Stats not found.")
        return
    stats = np.load(STATS_PATH)
    mean, std = stats["mean"], stats["std"]
    
    device = torch.device("cpu")
    
    # We use a specific seed (1000) to ensure we get a nice map with curves
    env = MetaDriveEnv(dict(
        use_render=False, # We don't need to see it, just record data
        manual_control=False,
        traffic_density=0.05, 
        start_seed=1000, 
        map="STO" # S=Straight, T=Turn, O=Roundabout (Good mix)
    ))
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    model = BCAgent(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. Run Showcase Episode
    print("Running showcase episode...")
    obs, info = env.reset()
    done = False
    
    # LOGGING LISTS
    log_steering = []
    log_throttle = []
    log_speed = []
    log_lateral_error = [] # Distance from center of lane
    
    while not done:
        norm_obs = (obs - mean) / std
        tensor_obs = torch.FloatTensor(norm_obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = model(tensor_obs).cpu().numpy()[0]
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # RECORD TELEMETRY
        log_steering.append(action[0])
        log_throttle.append(action[1])
        
        # MetaDrive vehicle info
        vehicle = env.vehicle
        log_speed.append(vehicle.speed)
        
        # Calculate approximate lateral error (if available in info, otherwise skip)
        # Note: MetaDrive doesn't always expose this easily in 'info', 
        # so we will focus on inputs/outputs.
        
    env.close()
    
    # 3. Generate Plots
    print(f"Episode finished. Steps: {len(log_speed)}. Generating plot...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    steps = range(len(log_speed))
    
    # PLOT A: Steering (The "Jitter" Check)
    ax1.plot(steps, log_steering, color='#3498db', label='Steering')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title("Steering Command (Smoothness Check)")
    ax1.set_ylabel("Steering [-1, 1]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PLOT B: Throttle/Brake (The "Logic" Check)
    # We color code: Green = Gas, Red = Brake
    ax2.plot(steps, log_throttle, color='gray', alpha=0.3) # Background trace
    ax2.fill_between(steps, log_throttle, 0, where=(np.array(log_throttle) >= 0), 
                     color='#2ecc71', alpha=0.5, label='Gas')
    ax2.fill_between(steps, log_throttle, 0, where=(np.array(log_throttle) < 0), 
                     color='#e74c3c', alpha=0.5, label='Brake')
    ax2.set_title("Pedal Input (Brake Injection Verification)")
    ax2.set_ylabel("Throttle / Brake")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # PLOT C: Vehicle Speed
    ax3.plot(steps, log_speed, color='purple', linewidth=2)
    ax3.set_title("Vehicle Speed Profile")
    ax3.set_ylabel("Speed (km/h or m/s)")
    ax3.set_xlabel("Simulation Steps")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PATH)
    print(f"📊 Telemetry Dashboard saved to: {SAVE_PLOT_PATH}")

if __name__ == "__main__":
    plot_telemetry()