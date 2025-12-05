import os
import numpy as np
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ============== CONFIGURATIONS =============
NUM_EPISODES_TO_RECORD = 5000
DATA_SAVE_PATH = "./file/expert_data/new_success_only/"
BUFFER_SIZE = 500_000 
EXPORT_NAME = "expert_metadrive_500k_5000eps_normalized" 
STATS_NAME = "expert_metadrive_500k_5000eps_normalized_stats.pkl"
TRAFFIC_DENSITY = 0.3

# NOISE SETTINGS (For recovery learning)
NOISE_PROBABILITY = 0.15 
STEERING_NOISE_STD = 0.3 

COLLECT_CONFIG = {
    "use_render": False,
    "manual_control": False,
    "start_seed": 0,
    "num_scenarios": 1000,
    "traffic_density": TRAFFIC_DENSITY,
    "out_of_road_penalty": 30.0,
    "crash_vehicle_penalty": 30.0,
    "crash_object_penalty": 30.0,
    "success_reward": 100.0,
    "use_lateral_reward": True,
    "window_size": (100, 100),
}

def make_env():
    return MetaDriveEnv(COLLECT_CONFIG)

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

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    inner_env = env.envs[0]

    # env = MetaDriveEnv(COLLECT_CONFIG)
    
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
            
            # Vary the seed per episode
            episode_seed = episode % COLLECT_CONFIG["num_scenarios"]
            obs = env.reset()
            
            # Create fresh expert policy for this episode's agent
            expert_policy = IDMPolicy(control_object=inner_env.agent, random_seed=episode_seed)
            expert_policy.reset()
            
            done = False
            episode_reward = 0
            episode_buffer = []

            while not done:
                # 1. GET EXPERT ACTION
                expert_action = get_expert_action(expert_policy)
                
                # 2. NOISE INJECTION (For Recovery)
                action_to_execute = expert_action.copy()
                if np.random.rand() < NOISE_PROBABILITY:
                    noise = np.random.normal(0, STEERING_NOISE_STD)
                    action_to_execute[0] = np.clip(action_to_execute[0] + noise, -1.0, 1.0)

                # 3. STEP
                # next_obs, reward, terminated, truncated, info = env.step(action_to_execute)
                next_obs, reward, dones, infos = env.step([action_to_execute])                
                
                # Extract scalars
                done = dones[0]
                info = infos[0]

                # Store in temporary buffer instead of main buffer
                episode_buffer.append({
                    "obs": obs.copy(),
                    "next_obs": next_obs.copy(),
                    "action": expert_action.copy(),
                    "reward": reward,
                    "done": done,
                    "info": info
                })

                obs = next_obs
                episode_reward += reward 
            # ==== END OF EPISODE ====

            if info.get("arrive_dest", False):
                # Success! Save all transitions from this episode
                for trans in episode_buffer:
                    replay_buffer.add(
                        trans["obs"],
                        trans["next_obs"],
                        trans["action"],
                        trans["reward"],
                        trans["done"],
                        [trans["info"]]
                    )
                    total_steps += 1
                episodes_collected += 1
                print(f"✓ Saved episode {episode} (Success)")
            else:
                # Failure - discarding episode
                print(f"✗ Discarded episode {episode} (Crash/Timeout)")

            total_reward += episode_reward
            
            if total_steps >= BUFFER_SIZE:
                print(f"Buffer full! Reached {total_steps:,} transitions.")
                break

    finally:
        # SAVE THE VECNORMALIZE STATS
        stats_path = os.path.join(DATA_SAVE_PATH, STATS_NAME)
        env.save(stats_path)
        print(f"\n✅ Saved VecNormalize statistics to: {stats_path}")

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

    print(f"\n{'='*60}")
    print("DATA VERIFICATION")
    print(f"{'='*60}")
    print(f"Total transitions: {total_steps:,}")
    print(f"Episodes collected: {episodes_collected}")
    print(f"\nObservation Statistics:")
    print(f"  Mean: {data['observations'].mean():.4f} (should be ~0 if normalized)")
    print(f"  Std:  {data['observations'].std():.4f} (should be ~1 if normalized)")
    print(f"  Min:  {data['observations'].min():.4f}")
    print(f"  Max:  {data['observations'].max():.4f}")
    print(f"\nAction Statistics:")
    print(f"  Mean: {data['actions'].mean(axis=0)} (steering, throttle)")
    print(f"  Std:  {data['actions'].std(axis=0)}")
    print(f"  Min:  {data['actions'].min(axis=0)}")
    print(f"  Max:  {data['actions'].max(axis=0)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    collect_data()