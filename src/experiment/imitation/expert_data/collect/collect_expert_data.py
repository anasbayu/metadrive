import os
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.buffers import ReplayBuffer

# ============== CONFIGURATIONS =============
NUM_EPISODES_TO_RECORD = 1200
DATA_SAVE_PATH = "./file/expert_data/"
BUFFER_SIZE = 500_000
EXPORT_NAME = "expert_metadrive_500k_clean"

TRAFFIC_DENSITY = 0.0  # ⚠️ Set to 0.0, 0.15, or 0.2 - then use SAME value in BC training!

COLLECT_CONFIG = {
    "use_render": False,
    "manual_control": False,
    "agent_policy": IDMPolicy,
    "num_scenarios": 100,
    "start_seed": np.random.randint(0, 10000),
    "traffic_density": TRAFFIC_DENSITY,
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True,
    "log_level": 50,
    "window_size": (100, 100),
}

def validate_data(data):
    """Validate collected data quality"""
    print(f"\n{'='*60}")
    print("DATA VALIDATION")
    print("="*60)
    
    obs_shape = data['observations'].shape
    act_shape = data['actions'].shape
    
    print(f"✓ Observations shape: {obs_shape}")
    print(f"✓ Actions shape: {act_shape}")
    print(f"✓ Rewards shape: {data['rewards'].shape}")
    print(f"✓ Dones shape: {data['dones'].shape}")
    
    # Check if arrays are empty
    if obs_shape[0] == 0 or act_shape[0] == 0:
        print(f"\n❌ ERROR: No data collected!")
        print(f"   Observations: {obs_shape[0]} samples")
        print(f"   Actions: {act_shape[0]} samples")
        return False
    
    print(f"\nAction Statistics:")
    print(f"  - Range: [{data['actions'].min():.4f}, {data['actions'].max():.4f}]")
    print(f"  - Mean: {data['actions'].mean():.4f}")
    print(f"  - Std: {data['actions'].std():.4f}")
    
    # Check for NaN or Inf
    has_nan_obs = np.isnan(data['observations']).any()
    has_nan_act = np.isnan(data['actions']).any()
    has_inf_obs = np.isinf(data['observations']).any()
    has_inf_act = np.isinf(data['actions']).any()
    
    if has_nan_obs or has_nan_act or has_inf_obs or has_inf_act:
        print(f"\n⚠️  WARNING: Data contains NaN or Inf values!")
        print(f"   Obs NaN: {has_nan_obs}, Inf: {has_inf_obs}")
        print(f"   Act NaN: {has_nan_act}, Inf: {has_inf_act}")
    else:
        print(f"\n✅ Data quality: GOOD (no NaN/Inf)")
    
    print(f"\nReward Statistics:")
    print(f"  - Total: {data['rewards'].sum():.2f}")
    print(f"  - Mean: {data['rewards'].mean():.4f}")
    print(f"  - Range: [{data['rewards'].min():.4f}, {data['rewards'].max():.4f}]")
    
    done_count = data['dones'].sum()
    print(f"\nEpisode Statistics:")
    print(f"  - Number of episodes: {int(done_count)}")
    print(f"  - Average episode length: {obs_shape[0] / max(done_count, 1):.1f} steps")
    
    print("="*60 + "\n")
    return True

def collect_data():
    """
    Collect CLEAN expert demonstrations using IDM policy.
    Suitable for BC training with imitation library.
    """
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    
    print("="*60)
    print("CLEAN EXPERT DATA COLLECTION")
    print("="*60)
    print(f"Target episodes: {NUM_EPISODES_TO_RECORD}")
    print(f"Buffer size: {BUFFER_SIZE:,}")
    print(f"Traffic density: {TRAFFIC_DENSITY}")
    print(f"Random seed: {COLLECT_CONFIG['start_seed']}")
    print("="*60 + "\n")
    
    # Create environment
    env = MetaDriveEnv(COLLECT_CONFIG)
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Action space bounds: {env.action_space.low} to {env.action_space.high}\n")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        BUFFER_SIZE,
        env.observation_space,
        env.action_space,
        device="cpu",
        n_envs=1
    )

    # Dummy action (will be ignored since we use expert policy)
    dummy_action = env.action_space.sample()

    total_steps = 0
    episodes_collected = 0
    total_reward = 0.0
    action_extraction_failures = 0
    
    print(f"Starting expert data collection...\n")
    
    try:
        for episode in range(NUM_EPISODES_TO_RECORD):
            # Progress update
            if (episode + 1) % 50 == 0 or episode == 0:
                buffer_pct = (total_steps / BUFFER_SIZE) * 100
                avg_reward = total_reward / max(episodes_collected, 1)
                print(f"Episode {episode + 1:4d} | "
                      f"Steps: {total_steps:7,}/{BUFFER_SIZE:,} ({buffer_pct:5.1f}%) | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Failures: {action_extraction_failures}")
            
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0

            while not terminated and not truncated:
                # Step with dummy action (expert policy controls the agent)
                next_obs, reward, terminated, truncated, info = env.step(dummy_action)
                
                # Extract expert's actual action - TRY MULTIPLE METHODS
                expert_action = None
                
                # Method 1: From agent's last_action attribute
                try:
                    expert_action_raw = env.agents["default_agent"].last_action
                    if expert_action_raw is not None:
                        expert_action = np.array(expert_action_raw, dtype=np.float32).flatten()
                except (KeyError, AttributeError) as e:
                    pass
                
                # Method 2: From info dict
                if expert_action is None:
                    try:
                        if "action" in info:
                            expert_action = np.array(info["action"], dtype=np.float32).flatten()
                    except Exception:
                        pass
                
                # Method 3: From agent's last_current_action
                if expert_action is None:
                    try:
                        expert_action_raw = env.agents["default_agent"].last_current_action
                        if expert_action_raw is not None:
                            expert_action = np.array(expert_action_raw, dtype=np.float32).flatten()
                    except (KeyError, AttributeError):
                        pass
                
                # If all methods failed
                if expert_action is None:
                    action_extraction_failures += 1
                    if action_extraction_failures <= 5:  # Only print first few
                        print(f"\n⚠️  Warning: Could not extract expert action (failure #{action_extraction_failures})")
                        print(f"    Available agent attributes: {dir(env.agents['default_agent'])[:10]}...")
                    break
                
                # Validate and fix action shape
                expected_shape = env.action_space.shape[0]
                if expert_action.shape[0] != expected_shape:
                    if episode == 0 and episode_steps == 0:
                        print(f"\n⚠️  Action shape mismatch detected!")
                        print(f"    Got: {expert_action.shape}, Expected: ({expected_shape},)")
                    
                    if expert_action.shape[0] == 1 and expected_shape == 2:
                        # Only steering, add moderate throttle
                        expert_action = np.array([expert_action[0], 0.5], dtype=np.float32)
                        if episode == 0 and episode_steps == 0:
                            print(f"    Fixed: Adding throttle=0.5 → shape {expert_action.shape}")
                    else:
                        action_extraction_failures += 1
                        if action_extraction_failures <= 5:
                            print(f"\n⚠️  Cannot fix action shape, skipping step")
                        continue
                
                # Clip to valid range
                expert_action = np.clip(expert_action, env.action_space.low, env.action_space.high)
                
                # Validate for NaN/Inf
                if np.isnan(expert_action).any() or np.isinf(expert_action).any():
                    action_extraction_failures += 1
                    if action_extraction_failures <= 5:
                        print(f"\n⚠️  Invalid action values (NaN/Inf), skipping step")
                    continue
                
                if np.isnan(next_obs).any() or np.isinf(next_obs).any():
                    action_extraction_failures += 1
                    if action_extraction_failures <= 5:
                        print(f"\n⚠️  Invalid observation (NaN/Inf), skipping step")
                    continue
                
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
                    buffer_pct = (total_steps / BUFFER_SIZE) * 100
                    print(f"\n✓ Buffer full! Collected {total_steps:,} transitions ({buffer_pct:.1f}%)")
                    print(f"  Completed {episode + 1} episodes")
                    terminated = True

            total_reward += episode_reward
            episodes_collected += 1
            
            if total_steps >= BUFFER_SIZE:
                break

    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    print(f"\n{'='*60}")
    print("Collection Complete")
    print("="*60)
    print(f"Total steps collected: {total_steps:,}")
    print(f"Episodes completed: {episodes_collected}/{NUM_EPISODES_TO_RECORD}")
    if episodes_collected > 0:
        print(f"Average episode reward: {total_reward / episodes_collected:.2f}")
    print(f"Action extraction failures: {action_extraction_failures}")
    print("="*60 + "\n")
    
    # Check if we actually collected data
    if total_steps == 0:
        print("❌ ERROR: No data was collected!")
        print("\n📝 Troubleshooting:")
        print("   1. Check that MetaDrive is properly installed")
        print("   2. Verify that IDMPolicy is working correctly")
        print("   3. Try running a simple MetaDrive example first")
        return
    
    # Prepare data for saving
    # CRITICAL FIX: When buffer is full, pos wraps to 0!
    # Use total_steps instead of replay_buffer.pos
    actual_size = min(total_steps, BUFFER_SIZE)
    print(f"Preparing to save {actual_size:,} transitions...\n")
    
    data = {
        "observations": replay_buffer.observations[:actual_size],
        "actions": replay_buffer.actions[:actual_size],
        "next_observations": replay_buffer.next_observations[:actual_size],
        "rewards": replay_buffer.rewards[:actual_size],
        "dones": replay_buffer.dones[:actual_size],
    }
    
    # Validate before saving
    if not validate_data(data):
        print("❌ Data validation failed! Not saving.")
        return
    
    # Save data
    save_file = os.path.join(DATA_SAVE_PATH, EXPORT_NAME)
    print(f"Saving data to {save_file}.npz ...")
    
    try:
        np.savez_compressed(save_file, **data)
        file_size_mb = os.path.getsize(f"{save_file}.npz") / (1024 * 1024)
        print(f"✓ Save complete! File size: {file_size_mb:.2f} MB")
        print(f"✓ Location: {save_file}.npz\n")
        
        # Print usage instructions
        print("="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"1. Update BC training script:")
        print(f"   EXPERT_DATA_PATH = '{save_file}.npz'")
        print(f"   TRAFFIC_DENSITY = {TRAFFIC_DENSITY}  # MUST MATCH!")
        print(f"\n2. Train BC:")
        print(f"   python train_bc_imitation.py")
        print(f"\n3. Verify BC success rate > 60%")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        raise

if __name__ == "__main__":
    collect_data()