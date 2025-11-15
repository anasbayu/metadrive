import os
import numpy as np
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.buffers import ReplayBuffer

# ============== CONFIGURATIONS =============
NUM_EPISODES_TO_RECORD = 100  
DATA_SAVE_PATH = "./file/expert_data/"
BUFFER_SIZE = 200_000  # number of data collected
EXPORT_NAME = "expert_metadrive_buffer_3"

COLLECT_CONFIG = {
    "use_render": False,     # Fast mode (no graphics)
    "manual_control": False, # Disable human control
    "agent_policy": IDMPolicy,  # Use IDM as the expert policy
    "num_scenarios": 100,
    "start_seed": 0,
    "traffic_density": 0.0,
    "log_level": 50,
    "window_size": (100, 100),
}

def collect_data():
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    
    env = MetaDriveEnv(COLLECT_CONFIG)
    obs_space = env.observation_space
    action_space = env.action_space
    
    replay_buffer = ReplayBuffer(
        BUFFER_SIZE,
        obs_space,
        action_space,
        device="cpu",
        n_envs=1
    )

    print("Initializing environment by calling reset...")
    obs, info = env.reset()

    # dummy action will be ignored since we use expert policy
    dummy_action = env.action_space.sample()

    total_steps = 0
    print(f"--- Starting Expert Data Collection for {NUM_EPISODES_TO_RECORD} episodes ---")
    
    for episode in range(NUM_EPISODES_TO_RECORD):
        if (episode + 1) % 10 == 0:
            print(f"--- Collecting Episode {episode + 1}/{NUM_EPISODES_TO_RECORD} ---")
            
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            # expert_action = expert_policy.act()
            # next_obs, reward, terminated, truncated, info = env.step(expert_action)

            next_obs, reward, terminated, truncated, info = env.step(dummy_action)
            # expert_action_list = info["action"]
            # expert_action_np = np.array(expert_action_list) # Convert to numpy array for sb3 buffer

            expert_action_tuple = env.agents["default_agent"].last_action
            expert_action = np.array(expert_action_tuple)

            replay_buffer.add(
                obs,
                next_obs,
                # expert_action_np, # Save the EXPERT action
                expert_action,
                reward,
                terminated,
                [info]
            )

            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if total_steps >= BUFFER_SIZE:
                print("Buffer is full. Stopping collection.")
                terminated = True # Break inner loop

        if total_steps >= BUFFER_SIZE:
            break # Break outer loop

    env.close()
    print(f"--- Collection Complete. Total Steps: {total_steps} ---")
    
    save_file = os.path.join(DATA_SAVE_PATH, EXPORT_NAME)
    print(f"Saving data to {save_file}.npz ...")
    
    # Save only the filled portion of the buffer
    data = {
        "observations": replay_buffer.observations[:replay_buffer.pos],
        "actions": replay_buffer.actions[:replay_buffer.pos],
        "next_observations": replay_buffer.next_observations[:replay_buffer.pos],
        "rewards": replay_buffer.rewards[:replay_buffer.pos],
        "dones": replay_buffer.dones[:replay_buffer.pos],
    }
    np.savez_compressed(save_file, **data)
    print("Save complete.")

if __name__ == "__main__":
    collect_data()