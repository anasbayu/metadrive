import os
import numpy as np
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.buffers import ReplayBuffer

# ============== CONFIGURATIONS =============
NUM_EPISODES_TO_RECORD = 1200
DATA_SAVE_PATH = "./file/expert_data/"
BUFFER_SIZE = 500_000 
EXPORT_NAME = "expert_metadrive_500k_1200eps_with_recovery_v3" 

# NOISE SETTINGS (For recovery learning)
NOISE_PROBABILITY = 0.15 
STEERING_NOISE_STD = 0.3 

COLLECT_CONFIG = {
    "use_render": False,
    "manual_control": False,
    "num_scenarios": 100,
    "start_seed": np.random.randint(0, 10000),
    "traffic_density": 0.15,
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True,
    "window_size": (100, 100),
}

def get_expert_action(policy):
    """
    EXTREMELY ROBUST Action Extractor.
    Forces the output to be [Steering, Acceleration].
    """
    # 1. Get raw action
    raw_action = policy.act(agent_id="default_agent")
    
    # 2. Extract specific action from tuple if needed
    # IDM often returns (action_array, info_dict)
    if isinstance(raw_action, tuple):
        # Check if the first element is the action array
        if hasattr(raw_action[0], '__iter__') or isinstance(raw_action[0], np.ndarray):
            action = raw_action[0]
        else:
            action = raw_action
    else:
        action = raw_action

    # 3. Flatten and standardize
    action = np.array(action, dtype=np.float32).flatten()
    
    # 4. THE FIX: Handle Missing Acceleration
    # If we only got 1 value (Steering), we MUST add Acceleration manually.
    if action.shape[0] == 1:
        # print(f"DEBUG: Found 1D action {action}. Appending throttle.") 
        # Create [Steering, 0.5] -> 0.5 means "Moderate Gas"
        action = np.array([action[0], 0.5], dtype=np.float32)
        
    # 5. Handle weird shapes (like empty arrays)
    elif action.shape[0] == 0:
         # Emergency fallback
         action = np.array([0.0, 0.5], dtype=np.float32)

    # 6. Clip to match MetaDrive space [-1, 1]
    action = np.clip(action, -1.0, 1.0)
    
    return action

def collect_data():
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    env = MetaDriveEnv(COLLECT_CONFIG)
    
    replay_buffer = ReplayBuffer(
        BUFFER_SIZE,
        env.observation_space,
        env.action_space,
        device="cpu",
        n_envs=1
    )

    total_steps = 0
    episodes_collected = 0
    total_reward = 0.0

    print(f"\n{'='*60}")
    print(f"Starting FIXED Data Collection")
    print(f"{'='*60}\n")
    
    try:
        for episode in range(NUM_EPISODES_TO_RECORD):
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{NUM_EPISODES_TO_RECORD} | Steps: {total_steps:,}")
            
            obs, info = env.reset()
            expert_policy = IDMPolicy(control_object=env.agent, random_seed=COLLECT_CONFIG['start_seed'])
            
            # Reset expert internal state
            expert_policy.reset()
            
            terminated = False
            truncated = False
            episode_reward = 0
            
            while not terminated and not truncated:
                # 1. ROBUSTLY GET EXPERT ACTION
                expert_action = get_expert_action(expert_policy)
                
                # 2. NOISE INJECTION (For Recovery)
                action_to_execute = expert_action.copy()
                if np.random.rand() < NOISE_PROBABILITY:
                    noise = np.random.normal(0, STEERING_NOISE_STD)
                    action_to_execute[0] = np.clip(action_to_execute[0] + noise, -1.0, 1.0)

                # 3. STEP
                next_obs, reward, terminated, truncated, info = env.step(action_to_execute)
                
                # 4. SAVE
                replay_buffer.add(
                    obs,
                    next_obs,
                    expert_action, # Save CLEAN action
                    reward,
                    terminated or truncated,
                    [info]
                )

                obs = next_obs
                episode_reward += reward
                total_steps += 1
                
                if total_steps >= BUFFER_SIZE:
                    break 

            total_reward += episode_reward
            episodes_collected += 1
            if total_steps >= BUFFER_SIZE: break

    finally:
        env.close()
    
    # Save
    data = {
        "observations": replay_buffer.observations[:total_steps],
        "actions": replay_buffer.actions[:total_steps],
        "next_observations": replay_buffer.next_observations[:total_steps],
        "rewards": replay_buffer.rewards[:total_steps],
        "dones": replay_buffer.dones[:total_steps],
    }
    
    save_file = os.path.join(DATA_SAVE_PATH, EXPORT_NAME)
    print(f"Saving to {save_file}.npz ...")
    np.savez_compressed(save_file, **data)
    print("Done!")

if __name__ == "__main__":
    collect_data()