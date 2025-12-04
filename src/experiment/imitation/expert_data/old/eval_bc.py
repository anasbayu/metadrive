import numpy as np
import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import os

# ================= CONFIGURATION =================
MODEL_PATH = "./file/model/bc_policy_imitation.zip"  # ✅ Updated for imitation library
EVAL_EPISODES = 50
RENDER = False  # Set to True to visualize (slower)

# 🔧 CRITICAL: Must match your BC training environment!
TRAFFIC_DENSITY = 0.0  # ⚠️ Set to match your data collection and BC training

EVAL_CONFIG = {
    "use_render": RENDER,
    "manual_control": False,
    "traffic_density": TRAFFIC_DENSITY,
    "num_scenarios": 100,
    "start_seed": 1000,  # Different from training (0) for generalization test
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True,
    "log_level": 50,
    "window_size": (1600, 1000) if RENDER else (100, 100),
}
# =================================================

def evaluate_bc_policy():
    """
    Evaluate BC policy trained with imitation library
    """
    print("="*60)
    print("🚗 BC POLICY EVALUATION")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Episodes: {EVAL_EPISODES}")
    print(f"Traffic Density: {TRAFFIC_DENSITY}")
    print(f"Render: {RENDER}")
    print("="*60 + "\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("\nPlease train BC first:")
        print("  python train_bc_imitation.py")
        return
    
    # Setup MetaDrive assets
    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
    
    # Create environment
    print("🌍 Creating evaluation environment...")
    env = MetaDriveEnv(EVAL_CONFIG)
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}\n")
    
    # Load BC policy
    print(f"📦 Loading BC policy from {MODEL_PATH}...")
    try:
        policy = ActorCriticPolicy.load(MODEL_PATH)
        print("✅ Policy loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        env.close()
        return
    
    # Evaluation metrics
    success_count = 0
    crash_count = 0
    out_of_road_count = 0
    timeout_count = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"{'='*60}")
    print(f"🏁 STARTING EVALUATION")
    print(f"{'='*60}\n")
    
    # Evaluation loop
    for episode in range(EVAL_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not terminated and not truncated:
            # Get action from BC policy (deterministic)
            action, _states = policy.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1
            
            # Optional: Display action in render mode
            if RENDER:
                env.render(text={
                    "Episode": f"{episode+1}/{EVAL_EPISODES}",
                    "Reward": f"{episode_reward:.1f}",
                    "Steering": f"{action[0]:.2f}",
                    "Throttle": f"{action[1]:.2f}"
                })
        
        # Store metrics
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Analyze episode outcome
        success = info.get("arrive_dest", False)
        crash = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        max_step = info.get("max_step", False)
        
        if success:
            success_count += 1
            result = "✅ SUCCESS"
        elif crash:
            crash_count += 1
            result = "💥 CRASH"
        elif out_of_road:
            out_of_road_count += 1
            result = "🛑 OUT-OF-ROAD"
        elif max_step:
            timeout_count += 1
            result = "⏰ TIMEOUT"
        else:
            result = "❓ UNKNOWN"
        
        print(f"Episode {episode+1:3d}/{EVAL_EPISODES} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Steps: {step:4d} | "
              f"{result}")
    
    env.close()
    
    # Calculate statistics
    success_rate = (success_count / EVAL_EPISODES) * 100
    crash_rate = (crash_count / EVAL_EPISODES) * 100
    out_of_road_rate = (out_of_road_count / EVAL_EPISODES) * 100
    timeout_rate = (timeout_count / EVAL_EPISODES) * 100
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    # Print final report
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"\n🎯 OUTCOMES:")
    print(f"   Success Rate:       {success_rate:6.2f}% ({success_count}/{EVAL_EPISODES})")
    print(f"   Crash Rate:         {crash_rate:6.2f}% ({crash_count}/{EVAL_EPISODES})")
    print(f"   Out-of-Road Rate:   {out_of_road_rate:6.2f}% ({out_of_road_count}/{EVAL_EPISODES})")
    print(f"   Timeout Rate:       {timeout_rate:6.2f}% ({timeout_count}/{EVAL_EPISODES})")
    
    print(f"\n💰 REWARDS:")
    print(f"   Average Reward:     {avg_reward:7.2f} ± {std_reward:.2f}")
    print(f"   Min Reward:         {np.min(total_rewards):7.2f}")
    print(f"   Max Reward:         {np.max(total_rewards):7.2f}")
    
    print(f"\n📏 EPISODE LENGTH:")
    print(f"   Average Length:     {avg_length:7.1f} steps")
    print(f"   Min Length:         {np.min(episode_lengths):4d} steps")
    print(f"   Max Length:         {np.max(episode_lengths):4d} steps")
    
    print(f"\n{'='*60}")
    print("🎓 PERFORMANCE ASSESSMENT")
    print("="*60)
    
    # Performance rating
    if success_rate >= 80:
        rating = "🌟 EXCELLENT"
        recommendation = "Outstanding! Ready for PPO fine-tuning."
    elif success_rate >= 60:
        rating = "✅ GOOD"
        recommendation = "Solid performance. Ready for PPO fine-tuning."
    elif success_rate >= 40:
        rating = "⚠️  FAIR"
        recommendation = "Acceptable. Can proceed to PPO, which should improve it."
    else:
        rating = "❌ POOR"
        recommendation = "Needs improvement. Consider:\n" + \
                        "       - Training BC for more epochs\n" + \
                        "       - Verifying traffic_density matches data\n" + \
                        "       - Collecting better quality expert data"
    
    print(f"\n   Rating: {rating}")
    print(f"   {recommendation}")
    
    print(f"\n{'='*60}")
    print("📝 NEXT STEPS")
    print("="*60)
    
    if success_rate >= 60:
        print("\n   ✅ BC policy is ready for PPO warmstart!")
        print(f"   📁 Model location: {MODEL_PATH}")
        print("\n   🚀 To proceed:")
        print("      1. Update train_ppo_imitation.py:")
        print(f"         BC_MODEL_PATH = '{MODEL_PATH}'")
        print(f"         TRAFFIC_DENSITY = {TRAFFIC_DENSITY}")
        print("      2. Run: python run_experiments_imitation.py")
    elif success_rate >= 40:
        print("\n   ⚠️  BC policy is acceptable but could be better.")
        print("\n   Options:")
        print("      1. Proceed to PPO (it will likely improve)")
        print("      2. Train BC longer (increase n_epochs to 150-200)")
        print("      3. Continue with current policy")
    else:
        print("\n   ❌ BC policy needs improvement before PPO.")
        print("\n   Troubleshooting:")
        print(f"      1. Verify TRAFFIC_DENSITY = {TRAFFIC_DENSITY} matches data collection")
        print("      2. Check expert data quality (avg reward should be >300)")
        print("      3. Try training BC for more epochs")
        print("      4. Consider re-collecting expert data")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    evaluate_bc_policy()