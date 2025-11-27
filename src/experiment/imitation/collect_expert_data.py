import os
import numpy as np
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.buffers import ReplayBuffer

# ============== CONFIGURATIONS =============
NUM_EPISODES_TO_RECORD = 1200
DATA_SAVE_PATH = "./file/expert_data/"
BUFFER_SIZE = 500_000  # number of data collected
EXPORT_NAME = "expert_metadrive_500k_1200eps"

COLLECT_CONFIG = {
    "use_render": False,     # Fast mode (no graphics)
    "manual_control": False, # Disable human control
    "agent_policy": IDMPolicy,  # Use IDM as the expert policy
    "num_scenarios": 100,
    "start_seed": np.random.randint(0, 10000),  # Random seed for diversity
    "traffic_density": 0.0,
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True,
    "log_level": 50,
    "window_size": (100, 100),
}

def validate_data(data):
    """Validate collected data and print statistics."""
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    obs_shape = data['observations'].shape
    act_shape = data['actions'].shape
    
    print(f"✓ Observations shape: {obs_shape}")
    print(f"✓ Actions shape: {act_shape}")
    print(f"✓ Next observations shape: {data['next_observations'].shape}")
    print(f"✓ Rewards shape: {data['rewards'].shape}")
    print(f"✓ Dones shape: {data['dones'].shape}")
    
    print(f"\nAction Statistics:")
    print(f"  - Range: [{data['actions'].min():.4f}, {data['actions'].max():.4f}]")
    print(f"  - Mean: {data['actions'].mean():.4f}")
    print(f"  - Std: {data['actions'].std():.4f}")
    
    print(f"\nReward Statistics:")
    print(f"  - Total: {data['rewards'].sum():.2f}")
    print(f"  - Mean: {data['rewards'].mean():.4f}")
    print(f"  - Range: [{data['rewards'].min():.4f}, {data['rewards'].max():.4f}]")
    
    done_count = data['dones'].sum()
    print(f"\nEpisode Statistics:")
    print(f"  - Number of episode endings: {int(done_count)}")
    print(f"  - Average episode length: {obs_shape[0] / max(done_count, 1):.1f} steps")
    
    print("="*60 + "\n")


def collect_data():
    """Collect expert driving data using MetaDrive IDMPolicy."""
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    
    print("="*60)
    print("METADRIVE EXPERT DATA COLLECTION")
    print("="*60)
    print(f"Target episodes: {NUM_EPISODES_TO_RECORD}")
    print(f"Buffer size: {BUFFER_SIZE:,}")
    print(f"Traffic density: {COLLECT_CONFIG['traffic_density']}")
    print(f"Random seed: {COLLECT_CONFIG['start_seed']}")
    print("="*60 + "\n")

    # Initialize environment
    env = MetaDriveEnv(COLLECT_CONFIG)
    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}\n")

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
    episodes_collected = 0
    total_reward = 0.0

    print(f"\n{'='*60}")
    print(f"Starting Expert Data Collection")
    print(f"{'='*60}\n")
    
    try:
        for episode in range(NUM_EPISODES_TO_RECORD):
            if (episode + 1) % 10 == 0:
                avg_reward = total_reward / (episode + 1) if episode > 0 else 0
                print(f"Episode {episode + 1}/{NUM_EPISODES_TO_RECORD} | "
                      f"Steps: {total_steps:,}/{BUFFER_SIZE:,} | "
                      f"Avg Reward: {avg_reward:.2f}")
                
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0

            while not terminated and not truncated:
                # Step with dummy action (expert policy controls the agent)
                next_obs, reward, terminated, truncated, info = env.step(dummy_action)
                
                # Extract the expert's actual action
                try:
                    expert_action_tuple = env.agents["default_agent"].last_action
                    expert_action = np.array(expert_action_tuple, dtype=np.float32)
                except (KeyError, AttributeError) as e:
                    print(f"⚠️ Warning: Could not extract expert action: {e}")
                    break
                
                # Determine if episode is done
                done = terminated or truncated
                
                # Add to replay buffer
                replay_buffer.add(
                    obs,
                    next_obs,
                    expert_action,
                    reward,
                    done,
                    [info]
                )

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Check if buffer is full
                if total_steps >= BUFFER_SIZE:
                    print(f"\n⚠️ Buffer full after {episode + 1}/{NUM_EPISODES_TO_RECORD} episodes")
                    print(f"   Collected {total_steps:,} steps")
                    terminated = True  # Break inner loop

            total_reward += episode_reward
            episodes_collected += 1
            
            if total_steps >= BUFFER_SIZE:
                break  # Break outer loop

    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        raise
    finally:
        env.close()
    
    print(f"\n{'='*60}")
    print(f"Collection Complete")
    print(f"{'='*60}")
    print(f"Total steps collected: {total_steps:,}")
    print(f"Episodes completed: {episodes_collected}/{NUM_EPISODES_TO_RECORD}")
    print(f"Average episode reward: {total_reward / max(episodes_collected, 1):.2f}")
    print(f"{'='*60}\n")
    
    # Prepare data for saving (only filled portion)
    actual_size = min(replay_buffer.pos, BUFFER_SIZE)
    print(f"Preparing to save {actual_size:,} transitions...\n")
    
    data = {
        "observations": replay_buffer.observations[:actual_size],
        "actions": replay_buffer.actions[:actual_size],
        "next_observations": replay_buffer.next_observations[:actual_size],
        "rewards": replay_buffer.rewards[:actual_size],
        "dones": replay_buffer.dones[:actual_size],
    }
    
    # Validate data before saving
    validate_data(data)
    
    # Save data
    save_file = os.path.join(DATA_SAVE_PATH, EXPORT_NAME)
    print(f"Saving data to {save_file}.npz ...")
    
    try:
        np.savez_compressed(save_file, **data)
        file_size_mb = os.path.getsize(f"{save_file}.npz") / (1024 * 1024)
        print(f"✓ Save complete! File size: {file_size_mb:.2f} MB")
        print(f"✓ Location: {save_file}.npz\n")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        raise

if __name__ == "__main__":
    collect_data()