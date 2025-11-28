import os
import numpy as np
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
import time

# ============== CONFIGURATIONS =============
DATA_TO_REPLAY = "file/expert_data/expert_metadrive_500k_1200eps_with_recovery_v3.npz"
# DATA_TO_REPLAY = "file/expert_data/expert_metadrive_buffer_3.npz"


REPLAY_CONFIG = {
    "use_render": True,
    "manual_control": False,
    "num_scenarios": 100,
    "start_seed": 0,
    "traffic_density": 0.0, 
    "log_level": 50,
    "window_size": (1200, 800)
}

def replay_data():
    print(f"--- Loading Expert Data for Replay from {DATA_TO_REPLAY} ---")
    
    if not os.path.exists(DATA_TO_REPLAY):
        print(f"Error: Data file not found at {DATA_TO_REPLAY}")
        print("Please run 'collect_expert_data.py' first.")
        return
        
    data = np.load(DATA_TO_REPLAY, allow_pickle=True)
    actions = data["actions"]
    dones = data["dones"].astype(bool).flatten()
    
    print(f"Loaded {len(actions)} actions to replay.")

    env = MetaDriveEnv(REPLAY_CONFIG)
    
    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
        
    print("\n--- Starting Replay ---")
    print("Environment will reset when a 'done' signal is encountered in the data.")
    
    obs, info = env.reset()
    
    for i in range(len(actions)):
        action = actions[i][0]
        is_done_from_data = dones[i]
        
        obs, reward, terminated, truncated, info = env.step(action)
    
        time.sleep(0.02)
        
        if terminated or truncated:
            print(f"  Step {i+1}: Episode finished. (Terminated: {terminated}, Truncated: {truncated})")
            print("  Resetting environment...")
            obs, info = env.reset()
            
            # Sanity check
            if not is_done_from_data:
                print(f"  Warning: Env terminated but data said 'not done'. Replay might be out of sync.")
                
        if i % 100 == 0:
             print(f"  Replaying step {i+1}/{len(actions)}...")

    env.close()
    print("\n--- Replay Complete ---")

if __name__ == "__main__":
    replay_data()