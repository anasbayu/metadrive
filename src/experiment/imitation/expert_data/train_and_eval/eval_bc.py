import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import gymnasium as gym

# ================= CONFIGURATION =================
MODEL_PATH = "./file/model/bc_policy_metadrive_v4" # Path to your saved model
EVAL_EPISODES = 50  # Run enough episodes to get a statistically valid score
TRAFFIC_DENSITY = 0.15 # Match your training density
# =================================================

def evaluate():
    # 1. Setup Environment (Use a different seed for testing!)
    env_config = {
        "use_render": False, # Set to True to watch it live
        "traffic_density": TRAFFIC_DENSITY,
        "num_scenarios": 100,
        "start_seed": 5000, # DIFFERENT seed from training (Gen. Capability)
        "random_lane_width": False,
        "random_agent_model": False,
    }
    env = MetaDriveEnv(env_config)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        policy = ActorCriticPolicy.load(MODEL_PATH, weights_only=False)
        print("✅ Loaded via SB3 .load()")
    except Exception:
        # --- OPTION B: If you saved using torch.save(policy) ---
        # We need to recreate the policy structure first!
        print("⚠️ Direct load failed. Reconstructing policy from state_dict...")
        
        # 1. Create a blank policy with the exact same architecture as training
        policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0, # Dummy LR
            net_arch=[256, 256] # MUST match your training script!
        )
        
        # 2. Load the weights
        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        
        # Handle case where checkpoint is a dict wrapping the state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            policy.load_state_dict(checkpoint) # It is likely the state_dict itself
        else:
            policy = checkpoint # It might be the full object after all
            
        print("✅ Reconstructed policy from weights.")

    print("Starting Evaluation...")
    
    success_count = 0
    crash_count = 0
    timeout_count = 0
    total_rewards = []

    for episode in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Predict action
            # BC policies in 'imitation' are usually standard SB3 policies
            action, _ = policy.predict(obs, deterministic=True)
            
            # --- DEBUG HACK: Force Forward Motion ---
            # If the model outputs < 0.1 throttle, force it to 0.3
            # Index 1 is throttle/brake in MetaDrive
            if action[1] < 0.1:
                action[1] = 0.3 
            # ----------------------------------------

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if env.config["use_render"]:
                env.render()

        # Analyze Result
        total_rewards.append(episode_reward)
        
        if info.get("arrive_dest"):
            print(f"Ep {episode+1}: ✅ Success! (Reward: {episode_reward:.2f})")
            success_count += 1
        elif info.get("crash"):
            print(f"Ep {episode+1}: ❌ CRASH. (Reward: {episode_reward:.2f})")
            crash_count += 1
        elif info.get("out_of_road"):
            print(f"Ep {episode+1}: ⚠️ Drove off road. (Reward: {episode_reward:.2f})")
            crash_count += 1
        else:
            print(f"Ep {episode+1}: 🕒 Timeout. (Reward: {episode_reward:.2f})")
            timeout_count += 1

    # 3. Final Report
    success_rate = (success_count / EVAL_EPISODES) * 100
    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT")
    print("="*30)
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Crash Rate:   {(crash_count / EVAL_EPISODES)*100:.1f}%")
    print(f"Timeout Rate: {(timeout_count / EVAL_EPISODES)*100:.1f}%")
    print(f"Avg Reward:   {np.mean(total_rewards):.2f}")
    print("="*30)
    
    env.close()

if __name__ == "__main__":
    evaluate()