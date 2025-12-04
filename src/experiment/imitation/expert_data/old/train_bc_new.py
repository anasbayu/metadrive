import os
import numpy as np
import torch
from functools import partial
from collections import defaultdict
import matplotlib.pyplot as plt

# Imitation library imports
from imitation.algorithms import bc
import imitation.data.types as im_types

# Stable Baselines3 imports
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# MetaDrive
from metadrive.envs.metadrive_env import MetaDriveEnv

# ============== CONFIGURATIONS =============
# Using 1000-scenario data for better generalization
EXPERT_DATA_PATH = "./file/expert_data/expert_metadrive_500k_1000scenarios_fixed.npz"  # ✅ 1000 scenarios

BC_POLICY_SAVE_PATH = "./file/model/bc_policy_imitation.zip"
EVAL_RESULTS_PATH = "./file/eval_results/bc_evaluation_imitation.npz"

# 🔧 CRITICAL: Set to match your expert data collection!
TRAFFIC_DENSITY = 0.0  # ⚠️ Change to 0.2 if your data was collected with traffic

BC_TRAIN_CONFIG = {
    "use_render": False,
    "manual_control": False,
    "log_level": 50,
    "num_scenarios": 100,
    "start_seed": 0,
    "traffic_density": TRAFFIC_DENSITY,  # ✅ Match expert data
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True,
    "window_size": (100, 100),
}

BC_EVAL_CONFIG = {
    "num_scenarios": 100,
    "start_seed": 0,  # ✅ CRITICAL: Use same scenarios as training! (0-99)
    "use_render": False,
    "manual_control": False,
    "log_level": 20,
    "traffic_density": TRAFFIC_DENSITY,  # ✅ Match training
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True,
    "window_size": (100, 100),
}

# BC Training hyperparameters
BC_HYPERPARAMS = {
    "n_epochs": 1,  # ✅ CRITICAL: Only 1 epoch! More causes catastrophic overfitting
    "batch_size": 256,  # Larger batch for stable learning
    "learning_rate": 3e-4,  # Standard BC learning rate
    "log_interval": 1,
}

def create_env_bc(config):
    """Create MetaDrive environment for BC training."""
    env = MetaDriveEnv(config)
    return env

def evaluate_bc_policy(policy, config, num_episodes=50, save_path=None):
    """
    🎯 Comprehensive BC evaluation
    """
    print(f"\n{'='*60}")
    print(f"🚗 STARTING BC POLICY EVALUATION")
    print(f"{'='*60}\n")
    
    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
        
    eval_env = create_env_bc(config)
    set_random_seed(12345)

    # Storage for metrics
    metrics = defaultdict(list)
    episode_data = []
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        actions_taken = []
        
        while not terminated and not truncated:
            action, _states = policy.predict(obs, deterministic=True)
            actions_taken.append(action)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            
        # Extract episode metrics
        success = info.get("arrive_dest", False)
        crash = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        max_step = info.get("max_step", False)
        
        # Calculate action smoothness
        action_variance = np.var(actions_taken, axis=0).mean() if len(actions_taken) > 1 else 0
        
        # Store metrics
        metrics["rewards"].append(episode_reward)
        metrics["lengths"].append(episode_length)
        metrics["successes"].append(int(success))
        metrics["crashes"].append(int(crash))
        metrics["out_of_roads"].append(int(out_of_road))
        metrics["max_steps"].append(int(max_step))
        metrics["action_variances"].append(action_variance)
        
        # Store full episode data
        episode_data.append({
            "episode": ep,
            "reward": episode_reward,
            "length": episode_length,
            "success": success,
            "crash": crash,
            "out_of_road": out_of_road,
            "action_variance": action_variance
        })
        
        # Print episode result
        status = "✅ SUCCESS" if success else "❌ FAILED"
        if crash:
            status += " (💥 CRASH)"
        if out_of_road:
            status += " (🛑 OUT-OF-ROAD)"
            
        print(f"Episode {ep+1:3d}/{num_episodes}: "
              f"Reward={episode_reward:7.2f} | "
              f"Length={episode_length:4d} | "
              f"{status}")
    
    eval_env.close()
    
    # Calculate summary statistics
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'='*60}\n")
    
    success_rate = np.mean(metrics["successes"]) * 100
    crash_rate = np.mean(metrics["crashes"]) * 100
    out_of_road_rate = np.mean(metrics["out_of_roads"]) * 100
    avg_reward = np.mean(metrics["rewards"])
    std_reward = np.std(metrics["rewards"])
    avg_length = np.mean(metrics["lengths"])
    avg_action_variance = np.mean(metrics["action_variances"])
    
    print(f"🎯 SUCCESS RATE:        {success_rate:6.2f}%")
    print(f"💥 COLLISION RATE:      {crash_rate:6.2f}%")
    print(f"🛑 OUT-OF-ROAD RATE:    {out_of_road_rate:6.2f}%")
    print(f"─" * 60)
    print(f"💰 AVERAGE REWARD:      {avg_reward:7.2f} ± {std_reward:.2f}")
    print(f"📏 AVERAGE LENGTH:      {avg_length:7.1f} steps")
    print(f"🎢 ACTION SMOOTHNESS:   {avg_action_variance:7.4f} (lower=smoother)")
    print(f"{'='*60}\n")
    
    # Performance rating
    print("📈 PERFORMANCE RATING:")
    if success_rate >= 80:
        rating = "🌟 EXCELLENT - Ready for PPO fine-tuning!"
    elif success_rate >= 60:
        rating = "✅ GOOD - Ready for PPO fine-tuning"
    elif success_rate >= 40:
        rating = "⚠️  FAIR - Consider more epochs or better data"
    else:
        rating = "❌ POOR - Check data quality or environment config"
    print(f"   {rating}\n")
    
    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            metrics=dict(metrics),
            episode_data=episode_data,
            summary={
                "success_rate": success_rate,
                "crash_rate": crash_rate,
                "out_of_road_rate": out_of_road_rate,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "avg_length": avg_length,
                "avg_action_variance": avg_action_variance
            }
        )
        print(f"💾 Results saved to: {save_path}\n")
    
    # Plot results
    plot_bc_evaluation(metrics, save_path)
    
    return metrics, success_rate

def plot_bc_evaluation(metrics, save_path=None):
    """📊 Visualize BC evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('BC Policy Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward Distribution
    axes[0, 0].hist(metrics["rewards"], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(metrics["rewards"]), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(metrics["rewards"]):.2f}')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Success/Failure Pie Chart
    labels = ['Success', 'Crash', 'Out-of-Road', 'Timeout']
    sizes = [
        sum(metrics["successes"]),
        sum(metrics["crashes"]),
        sum(metrics["out_of_roads"]),
        sum(metrics["max_steps"])
    ]
    colors = ['#06A77D', '#D81159', '#F24236', '#FFA500']
    axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Episode Outcomes')
    
    # Plot 3: Episode Lengths
    axes[0, 2].plot(metrics["lengths"], color='#8338EC', linewidth=1.5)
    axes[0, 2].axhline(np.mean(metrics["lengths"]), color='red', linestyle='--',
                       label=f'Mean: {np.mean(metrics["lengths"]):.1f}')
    axes[0, 2].set_title('Episode Length Over Time')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Plot 4: Reward Over Episodes
    axes[1, 0].plot(metrics["rewards"], color='#2E86AB', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title('Reward per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Reward')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: Action Smoothness
    axes[1, 1].plot(metrics["action_variances"], color='#FF6B35', linewidth=1.5)
    axes[1, 1].axhline(np.mean(metrics["action_variances"]), color='red', linestyle='--',
                       label=f'Mean: {np.mean(metrics["action_variances"]):.4f}')
    axes[1, 1].set_title('Action Variance (Smoothness)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Action Variance')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Success Rate Rolling Average
    window = 10
    rolling_success = np.convolve(metrics["successes"], 
                                   np.ones(window)/window, mode='valid')
    axes[1, 2].plot(rolling_success * 100, color='#06A77D', linewidth=2)
    axes[1, 2].set_title(f'Success Rate (Rolling Avg, window={window})')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Success Rate (%)')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plot_path = save_path.replace('.npz', '_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plots saved to: {plot_path}")
    
    plt.show()

def main():
    """Main BC training and evaluation pipeline"""
    print("="*60)
    print("🚗 BC TRAINING PIPELINE - Imitation Library")
    print("="*60)
    print(f"Traffic Density: {TRAFFIC_DENSITY}")
    print("="*60)
    
    # Check if expert data exists
    if not os.path.exists(EXPERT_DATA_PATH):
        print(f"\n❌ Error: Expert data not found at {EXPERT_DATA_PATH}")
        print("Please run data collection script first.")
        return

    # Load expert data
    print(f"\n📂 Loading expert data from {EXPERT_DATA_PATH}...")
    data = np.load(EXPERT_DATA_PATH, allow_pickle=True)
    
    # Handle observations - squeeze if needed
    obs = data["observations"]
    actions = data["actions"]
    
    if obs.ndim == 3:
        print(f"   ⚠️  Squeezing observations from {obs.shape} to 2D")
        obs = obs.squeeze(axis=1)
    
    if actions.ndim == 3:
        print(f"   ⚠️  Squeezing actions from {actions.shape} to 2D")
        actions = actions.squeeze(axis=1)
    
    num_transitions = len(obs)
    
    # CRITICAL: Ensure contiguous arrays for imitation library
    obs = np.ascontiguousarray(obs)
    actions = np.ascontiguousarray(actions)
    
    # Check data quality
    print(f"\n✅ Loaded {num_transitions:,} expert transitions")
    print(f"   - Observations shape: {obs.shape}")
    print(f"   - Actions shape: {actions.shape}")
    print(f"   - Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   - Actions mean: {actions.mean(axis=0)}")
    print(f"   - Actions std: {actions.std(axis=0)}")
    
    if "rewards" in data:
        rewards_mean = data['rewards'].mean()
        print(f"   - Rewards mean: {rewards_mean:.2f}")
        
        # CRITICAL CHECK: If expert reward is high but BC fails, something is wrong
        if rewards_mean < 100:
            print(f"\n   ⚠️  WARNING: Expert reward is very low ({rewards_mean:.2f})!")
            print(f"       This suggests data quality issues.")
            print(f"       Expected: ~377 (from visualization)")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Create infos if missing
    if "infos" in data:
        infos = data["infos"]
    else:
        print("   ⚠️  'infos' field not found. Creating dummy infos.")
        infos = np.array([{} for _ in range(num_transitions)])

    dones = data["dones"].astype(bool).flatten()
    
    # Handle next_observations
    if "next_observations" in data:
        next_obs = data["next_observations"]
        if next_obs.ndim == 3:
            next_obs = next_obs.squeeze(axis=1)
        next_obs = np.ascontiguousarray(next_obs)
    else:
        print("   ⚠️  'next_observations' not found, using observations")
        next_obs = obs.copy()

    # Create Transitions object
    print("\n📦 Creating imitation Transitions object...")
    expert_data = im_types.Transitions(
        obs=obs,
        acts=actions,
        next_obs=next_obs,
        dones=dones,
        infos=infos
    )
    
    print(f"   ✅ Transitions object created with {len(expert_data)} samples")

    # Create environment for BC
    print("\n🌍 Creating training environment...")
    venv = DummyVecEnv([partial(create_env_bc, BC_TRAIN_CONFIG)])
    print(f"   Observation space: {venv.observation_space}")
    print(f"   Action space: {venv.action_space}")

    # Setup BC trainer
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    rng = np.random.default_rng(0)
    
    print("\n🧠 Initializing BC trainer...")
    policy = ActorCriticPolicy(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        lr_schedule=lambda _: BC_HYPERPARAMS["learning_rate"],
        **policy_kwargs
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    print(f"   Network architecture: {policy_kwargs['net_arch']}")

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_data,
        policy=policy,
        rng=rng,
        device=device,
        batch_size=BC_HYPERPARAMS["batch_size"]
    )

    # Train BC
    print(f"\n{'='*60}")
    print(f"🏋️  Starting BC training...")
    print(f"{'='*60}")
    print(f"   Epochs: {BC_HYPERPARAMS['n_epochs']}")
    print(f"   Batch size: {BC_HYPERPARAMS['batch_size']}")
    print(f"   Learning rate: {BC_HYPERPARAMS['learning_rate']}")
    print(f"   Device: {device}")
    print(f"{'='*60}\n")
    
    bc_trainer.train(
        n_epochs=BC_HYPERPARAMS["n_epochs"],
        log_interval=BC_HYPERPARAMS["log_interval"]
    )
    print("\n✅ BC Training Complete!")

    # Save policy
    print(f"\n💾 Saving policy to {BC_POLICY_SAVE_PATH}...")
    os.makedirs(os.path.dirname(BC_POLICY_SAVE_PATH), exist_ok=True)
    bc_trainer.policy.save(BC_POLICY_SAVE_PATH)
    print("✅ Policy saved!")
    
    # Comprehensive evaluation
    print("\n" + "="*60)
    print("📊 STARTING EVALUATION PHASE")
    print("="*60)
    
    metrics, success_rate = evaluate_bc_policy(
        bc_trainer.policy, 
        BC_EVAL_CONFIG, 
        num_episodes=50,
        save_path=EVAL_RESULTS_PATH
    )
    
    venv.close()
    
    # Final recommendations
    print("\n" + "="*60)
    print("🎉 BC PIPELINE COMPLETE!")
    print("="*60)
    print("\n📝 NEXT STEPS:")
    if success_rate >= 60:
        print("   ✅ Success rate is good! Ready for PPO fine-tuning")
        print(f"   📁 BC Policy saved at: {BC_POLICY_SAVE_PATH}")
        print("   🚀 Next: Run PPO training with BC warmstart")
    elif success_rate >= 40:
        print("   ⚠️  Success rate is fair. Options:")
        print("      1. Train BC for more epochs")
        print("      2. Proceed to PPO (it might still improve)")
        print("      3. Collect more/better expert data")
    else:
        print("   ❌ Success rate is poor. Recommended actions:")
        print("      1. Check TRAFFIC_DENSITY matches data collection")
        print(f"         Current: {TRAFFIC_DENSITY}")
        print("      2. Verify expert data quality")
        print("      3. Consider re-collecting data")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()