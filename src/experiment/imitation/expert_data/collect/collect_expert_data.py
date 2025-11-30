import os
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from stable_baselines3.common.buffers import ReplayBuffer

# ============== CONFIGURATIONS =============
NUM_EPISODES_TO_RECORD = 1200
DATA_SAVE_PATH = "./file/expert_data/"
BUFFER_SIZE = 500_000
EXPORT_NAME = "expert_metadrive_500k_noisy_v5"

TRAFFIC_DENSITY = 0.15 

COLLECT_CONFIG = {
    "use_render": False,
    "manual_control": False,
    "agent_policy": IDMPolicy,
    "num_scenarios": 1000,
    "start_seed": 1000,
    "traffic_density": TRAFFIC_DENSITY,
    # Penalties don't matter for collection, only for the expert's internal logic
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "log_level": 50,
}

def collect_data():
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    env = MetaDriveEnv(COLLECT_CONFIG)
    
    # Init Buffer
    replay_buffer = ReplayBuffer(
        BUFFER_SIZE,
        env.observation_space,
        env.action_space,
        device="cpu",
        n_envs=1
    )

    dummy_action = np.zeros(env.action_space.shape) # Passive action
    total_steps = 0
    episodes_saved = 0
    
    print(f"Start Collection | Target: {BUFFER_SIZE} steps or {NUM_EPISODES_TO_RECORD} eps")
    
    for episode_idx in range(NUM_EPISODES_TO_RECORD):

        reset_return = env.reset()
        if isinstance(reset_return, tuple):
            obs, info = reset_return  # Gymnasium / New MetaDrive
        else:
            obs, info = reset_return, {}  # Gym / Old MetaDrive
        
        # Temp storage for THIS episode
        episode_transitions = []
        episode_reward = 0

        done = False
        truncated = False
        
        policy = env.engine.get_policy(env.vehicle.id)  # Access the IDM policy

        while not (done or truncated):
            # Get Expert Action from IDM
            raw_action = policy.act(agent_id=env.vehicle.id)
            expert_action = np.array(raw_action, dtype=np.float32).flatten()

            expert_action = np.clip(expert_action, -1.0, 1.0)

            # check for NaNs just in case
            if np.isnan(expert_action).any():
                print(f"⚠️ NaN Action Detected. Skipping step.")
                # Step the env anyway to keep physics running, but DO NOT SAVE.
                env.step(dummy_action)
                continue

            # Copy the expert action and add noise for diversity
            action_to_take = expert_action.copy()
            is_noise_step = False

            # 20% of the time, add noise to steering (index 0)
            if np.random.rand() < 0.20:
                noise = np.random.normal(0, 0.05) # 0.05 is the "Wobble Factor"
                action_to_take[0] = np.clip(action_to_take[0] + noise, -1.0, 1.0)
                is_noise_step = True

            # Take Step with Noisy Expert Action (20% chance of noise on steering)
            step_return = env.step(action_to_take)

            if len(step_return) == 5:
                # Gymnasium API: next_obs, reward, terminated, truncated, info
                next_obs, reward, terminated, truncated, info = step_return
                done = terminated or truncated
            elif len(step_return) == 4:
                # Old Gym API: next_obs, reward, done, info
                next_obs, reward, done, info = step_return
                truncated = False
            else:
                raise ValueError(f"Unexpected step return length: {len(step_return)}")

            
            if not is_noise_step:
                episode_transitions.append({
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": expert_action,
                    "reward": reward,
                    "done": done,
                    "info": info
                })
            
            obs = next_obs
            episode_reward += reward

        # ==== END OF EPISODE ====
        # Only save if episode Success
        if info.get("arrive_dest", False):
            # Commit this episode to the main buffer
            for trans in episode_transitions:
                replay_buffer.add(
                    trans["obs"],
                    trans["next_obs"],
                    trans["action"],
                    trans["reward"],
                    trans["done"],
                    [trans["info"]]
                )
                total_steps += 1
            
            episodes_saved += 1
            if episodes_saved % 50 == 0:
                print(f"Saved Ep {episodes_saved} | Steps: {total_steps} | Reward: {episode_reward:.2f}")
        else:
            # Discard failed episode
            print(f"Discarded Ep {episode_idx} (Crash/Timeout)")
            pass

        if total_steps >= BUFFER_SIZE:
            print("Buffer Full!")
            break

    env.close()

    # ============== SAVING ==============
    if total_steps == 0:
        print("❌ Error: No successful episodes collected.")
        return

    print(f"Saving {total_steps} transitions...")
    
    # Validating data before save
    print(f"Action Stats: Mean={replay_buffer.actions[:total_steps].mean():.3f}, "
          f"Min={replay_buffer.actions[:total_steps].min():.3f}, "
          f"Max={replay_buffer.actions[:total_steps].max():.3f}")


    # ==========================================
    # FINAL SANITY CHECK BEFORE SAVING
    # ==========================================
    print("\n🔍 Running Final Data Integrity Check...")
    
    # Since SB3 ReplayBuffer adds Num_Envs dimension, and we only need (N, 2) which is (N, Steering+Throttle)
    # We will need to squeeze out the extra dimension
    # 1. Extract and Squeeze (Remove the '1' middle dimension)
    # SB3 Buffer shape is (Step, Env, Dim). We want (Step, Dim).
    valid_obs = replay_buffer.observations[:total_steps, 0, :]
    valid_actions = replay_buffer.actions[:total_steps, 0, :]
    valid_next_obs = replay_buffer.next_observations[:total_steps, 0, :]
    valid_rewards = replay_buffer.rewards[:total_steps, 0] # Becomes 1D array
    valid_dones = replay_buffer.dones[:total_steps, 0]     # Becomes 1D array

    # Check Action Shape
    # Now valid_actions should be (N, 2)
    if valid_actions.ndim != 2 or valid_actions.shape[1] != 2:
        print(f"❌ FATAL ERROR: Bad Action Shape!")
        print(f"   Expected: (N, 2)")
        print(f"   Got:      {valid_actions.shape}")
        return # Stop execution

    # Check for 'Flat' Throttle
    throttle_std = np.std(valid_actions[:, 1])
    if throttle_std < 0.001:
        print(f"⚠️  WARNING: Throttle variance is near zero ({throttle_std:.5f})!")
    else:
        print(f"✓ Throttle Variance: {throttle_std:.4f} (Good)")

    print("✅ Integrity Check Passed: Data is Clean.")
    
    # ==========================================
    # SAVE EXECUTION
    # ==========================================
    save_path = os.path.join(DATA_SAVE_PATH, EXPORT_NAME)
    print(f"Saving to {save_path}.npz ...")
    
    np.savez_compressed(
        save_path,
        observations=valid_obs,
        actions=valid_actions,
        next_observations=valid_next_obs,
        rewards=valid_rewards,
        dones=valid_dones,
    )
    print(f"✅ Saved to {save_path}.npz")

if __name__ == "__main__":
    collect_data()