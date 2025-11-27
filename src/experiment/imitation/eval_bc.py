import numpy as np
import torch
import torch.nn as nn
from metadrive.envs.metadrive_env import MetaDriveEnv
import os

# ================= CONFIGURATION =================
MODEL_PATH = "./file/models/bc_agent_forced2d.pth"
STATS_PATH = "./file/models/normalization_stats.npz"
EVAL_EPISODES = 20
RENDER = True  # Set to False for faster evaluation
# =================================================

# Must match the Training Class EXACTLY
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

def evaluate_agent():
    # 1. Load Normalization Statistics
    print(f"Loading normalization stats from {STATS_PATH}...")
    if not os.path.exists(STATS_PATH):
        print("❌ Error: Normalization stats not found! Agent will fail.")
        return
        
    stats = np.load(STATS_PATH)
    mean = stats["mean"]
    std = stats["std"]
    print("✅ Stats loaded.")

    # 2. Setup Environment
    # Note: We use 0 traffic to match your specific training optimization
    env = MetaDriveEnv(dict(
        use_render=RENDER,
        manual_control=False,
        traffic_density=0.0, 
        num_scenarios=100,
        start_seed=1000, # Use different seeds than training (0-999) to test generalization
        random_lane_width=True,
        random_lane_num=True,
        window_size=(1000, 800)
    ))
    
    # 3. Initialize Model
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    device = torch.device("cpu") # CPU is usually fine for single-agent inference
    model = BCAgent(obs_dim, act_dim).to(device)
    
    print(f"Loading model weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # CRITICAL: Disables Dropout for deterministic driving
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 4. Evaluation Loop
    success_count = 0
    crash_count = 0
    out_of_road_count = 0
    total_rewards = []
    
    print(f"\n{'='*40}")
    print(f"STARTING EVALUATION ({EVAL_EPISODES} Episodes)")
    print(f"{'='*40}")

    for episode in range(EVAL_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # A. Normalize Observation
            norm_obs = (obs - mean) / std
            
            # B. Inference
            tensor_obs = torch.FloatTensor(norm_obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(tensor_obs).cpu().numpy()[0]
            
            # C. Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            if RENDER:
                env.render(text={"Action": f"Str: {action[0]:.2f} | Acc: {action[1]:.2f}"})

        total_rewards.append(episode_reward)
        
        # Analyze Result
        result = "UNKNOWN"
        if info.get("arrive_dest"):
            success_count += 1
            result = "✅ SUCCESS"
        elif info.get("crash"):
            crash_count += 1
            result = "💥 CRASH"
        elif info.get("out_of_road"):
            out_of_road_count += 1
            result = "🚫 OUT OF ROAD"
        elif info.get("max_step"):
             result = "⏰ TIMEOUT"
             
        print(f"Ep {episode+1}/{EVAL_EPISODES} | Reward: {episode_reward:.2f} | Steps: {step} | {result}")

    env.close()
    
    # 5. Final Report
    success_rate = (success_count / EVAL_EPISODES) * 100
    avg_reward = np.mean(total_rewards)
    
    print(f"\n{'='*40}")
    print("FINAL EVALUATION REPORT")
    print(f"{'='*40}")
    print(f"Success Rate:    {success_rate:.2f}%")
    print(f"Average Reward:  {avg_reward:.2f}")
    print(f"Crash Rate:      {(crash_count/EVAL_EPISODES)*100:.1f}%")
    print(f"Out of Road:     {(out_of_road_count/EVAL_EPISODES)*100:.1f}%")
    print(f"{'='*40}")

    if success_rate > 50:
        print("🚀 READY FOR PPO: This agent is good enough to initialize RL training.")
    else:
        print("⚠️ NEEDS WORK: Check data balancing or training epochs.")

if __name__ == "__main__":
    evaluate_agent()