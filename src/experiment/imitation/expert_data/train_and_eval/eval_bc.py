import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
MODEL_PATH = "./file/model/new_v3/bc_leakyppo_success_only.zip"
STATS_PATH = "./file/expert_data/new_success_only/expert_metadrive_500k_5000eps_normalized_stats.pkl"
PLOT_SAVE_PATH = "./file/model/new_v3/"
EVAL_EPISODES = 100  # More episodes for better statistics
TRAFFIC_DENSITY = 0.3  # Match your training density
NET_ARCH = [256, 256, 256]  # Match your BC architecture (wide for PPO, deep for LeakyPPO)
# =================================================

def create_eval_env():
    """Create evaluation environment with VecNormalize"""
    env_config = {
        "use_render": False,
        "traffic_density": TRAFFIC_DENSITY,
        "num_scenarios": 100,
        "start_seed": 1000,  # Match your BC evaluation
        "random_lane_width": False,
        "random_agent_model": False,
        "horizon": 1000,
    }
    
    def make_env():
        return MetaDriveEnv(env_config)
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats
    try:
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
        print(f"✓ Loaded VecNormalize stats from {STATS_PATH}")
    except Exception as e:
        print(f"⚠️ Could not load VecNormalize stats: {e}")
        print("Continuing without normalization...")
    
    return env

def load_bc_policy(env):
    """Load BC policy from checkpoint"""
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        policy = ActorCriticPolicy.load(MODEL_PATH, weights_only=False)
        print("✅ Loaded via ActorCriticPolicy.load()")
        return policy
    except Exception as e:
        print(f"⚠️ Direct load failed: {e}")
        print("Reconstructing policy from state_dict...")
        
        # Create blank policy with matching architecture
        policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,
            net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)
        )
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            policy.load_state_dict(checkpoint)
        else:
            policy = checkpoint
            
        print("✅ Reconstructed policy from weights.")
        return policy

def plot_diagnostics(results):
    """Create diagnostic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Reward Distribution
    ax = axes[0, 0]
    ax.hist(results['rewards'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(results['rewards']), color='red', linestyle='--', 
               label=f"Mean: {np.mean(results['rewards']):.1f}")
    ax.axvline(np.median(results['rewards']), color='green', linestyle='--',
               label=f"Median: {np.median(results['rewards']):.1f}")
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Length Distribution
    ax = axes[0, 1]
    ax.hist(results['lengths'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(results['lengths']), color='red', linestyle='--',
               label=f"Mean: {np.mean(results['lengths']):.0f}")
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Outcome Breakdown
    ax = axes[1, 0]
    outcomes = ['Success', 'Crash', 'Out of Road', 'Timeout']
    counts = [
        results['success_count'],
        results['crash_count'],
        results['out_of_road_count'],
        results['timeout_count']
    ]
    colors = ['green', 'red', 'orange', 'gray']
    bars = ax.bar(outcomes, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Episode Outcomes')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/EVAL_EPISODES*100:.1f}%)',
                ha='center', va='bottom')
    
    # 4. Reward vs Episode Length Scatter
    ax = axes[1, 1]
    successes = np.array(results['successes'])  # Convert to numpy array first
    success_mask = successes
    fail_mask = ~success_mask
    
    if np.any(success_mask):
        ax.scatter(np.array(results['lengths'])[success_mask], 
                  np.array(results['rewards'])[success_mask],
                  c='green', alpha=0.6, label='Success', s=50)
    if np.any(fail_mask):
        ax.scatter(np.array(results['lengths'])[fail_mask], 
                  np.array(results['rewards'])[fail_mask],
                  c='red', alpha=0.6, label='Failure', s=50)
    
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward vs Length (Success/Failure)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH + 'bc_evaluation_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"📊 Diagnostic plots saved to: {PLOT_SAVE_PATH}bc_evaluation_diagnostics.png")
    plt.close()

def evaluate():
    # Setup
    env = create_eval_env()
    policy = load_bc_policy(env)
    
    print("\n" + "="*60)
    print("STARTING DETAILED BC EVALUATION")
    print("="*60)
    print(f"Episodes: {EVAL_EPISODES}")
    print(f"Traffic Density: {TRAFFIC_DENSITY}")
    print(f"Architecture: {NET_ARCH}")
    print("="*60 + "\n")
    
    # Tracking variables
    results = {
        'rewards': [],
        'lengths': [],
        'successes': [],
        'success_count': 0,
        'crash_count': 0,
        'out_of_road_count': 0,
        'timeout_count': 0,
        'failure_times': []  # When did failures occur?
    }
    
    # Run evaluation
    for episode in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Predict action
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            # Check if done
            if done[0]:
                break
        
        # Extract info
        info_dict = info[0]
        results['rewards'].append(episode_reward)
        results['lengths'].append(steps)
        
        # Classify outcome
        is_success = info_dict.get("arrive_dest", False)
        results['successes'].append(is_success)
        
        if is_success:
            results['success_count'] += 1
            status = "✅ SUCCESS"
            color = "\033[92m"  # Green
        elif info_dict.get("crash"):
            results['crash_count'] += 1
            results['failure_times'].append(steps)
            status = "❌ CRASH"
            color = "\033[91m"  # Red
        elif info_dict.get("out_of_road"):
            results['out_of_road_count'] += 1
            results['failure_times'].append(steps)
            status = "⚠️ OFF ROAD"
            color = "\033[93m"  # Yellow
        else:
            results['timeout_count'] += 1
            status = "🕒 TIMEOUT"
            color = "\033[90m"  # Gray
        
        print(f"{color}Ep {episode+1:3d}/{EVAL_EPISODES}: {status:12s} | "
              f"Reward: {episode_reward:6.1f} | Steps: {steps:4d}\033[0m")
    
    env.close()
    
    # ========================================
    # COMPREHENSIVE DIAGNOSTIC REPORT
    # ========================================
    
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    
    # Overall Performance
    success_rate = (results['success_count'] / EVAL_EPISODES) * 100
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Success Rate:     {success_rate:5.1f}% ({results['success_count']}/{EVAL_EPISODES})")
    print(f"  Crash Rate:       {(results['crash_count']/EVAL_EPISODES)*100:5.1f}% ({results['crash_count']}/{EVAL_EPISODES})")
    print(f"  Off Road Rate:    {(results['out_of_road_count']/EVAL_EPISODES)*100:5.1f}% ({results['out_of_road_count']}/{EVAL_EPISODES})")
    print(f"  Timeout Rate:     {(results['timeout_count']/EVAL_EPISODES)*100:5.1f}% ({results['timeout_count']}/{EVAL_EPISODES})")
    
    # Reward Statistics
    print(f"\n💰 REWARD STATISTICS:")
    print(f"  Mean:             {np.mean(results['rewards']):6.1f}")
    print(f"  Std Dev:          {np.std(results['rewards']):6.1f}")
    print(f"  Median:           {np.median(results['rewards']):6.1f}")
    print(f"  Min:              {np.min(results['rewards']):6.1f}")
    print(f"  Max:              {np.max(results['rewards']):6.1f}")
    print(f"  25th Percentile:  {np.percentile(results['rewards'], 25):6.1f}")
    print(f"  75th Percentile:  {np.percentile(results['rewards'], 75):6.1f}")
    
    # Episode Length Statistics
    print(f"\n📏 EPISODE LENGTH STATISTICS:")
    print(f"  Mean:             {np.mean(results['lengths']):6.0f} steps")
    print(f"  Std Dev:          {np.std(results['lengths']):6.0f} steps")
    print(f"  Median:           {np.median(results['lengths']):6.0f} steps")
    print(f"  Min:              {np.min(results['lengths']):6.0f} steps")
    print(f"  Max:              {np.max(results['lengths']):6.0f} steps")
    
    # Failure Analysis
    if results['failure_times']:
        print(f"\n⚠️ FAILURE TIMING ANALYSIS:")
        print(f"  Mean Failure Time: {np.mean(results['failure_times']):6.0f} steps")
        print(f"  Median Failure:    {np.median(results['failure_times']):6.0f} steps")
        
        early_failures = sum(1 for t in results['failure_times'] if t < 200)
        mid_failures = sum(1 for t in results['failure_times'] if 200 <= t < 600)
        late_failures = sum(1 for t in results['failure_times'] if t >= 600)
        
        total_failures = len(results['failure_times'])
        print(f"  Early (<200):      {early_failures:3d} ({early_failures/total_failures*100:5.1f}%)")
        print(f"  Mid (200-600):     {mid_failures:3d} ({mid_failures/total_failures*100:5.1f}%)")
        print(f"  Late (>600):       {late_failures:3d} ({late_failures/total_failures*100:5.1f}%)")
    
    # Comparison to Expert
    expert_reward = 395.9  # From your expert data collection
    bc_performance_pct = (np.mean(results['rewards']) / expert_reward) * 100
    
    print(f"\n🎯 COMPARISON TO EXPERT:")
    print(f"  Expert Reward:     {expert_reward:6.1f}")
    print(f"  BC Reward:         {np.mean(results['rewards']):6.1f}")
    print(f"  BC Performance:    {bc_performance_pct:5.1f}% of expert")
    
    # Decision Recommendation
    print(f"\n🔍 RECOMMENDATION:")
    if success_rate < 20:
        print("  ⛔ BC performance is TOO LOW (<20% success)")
        print("  → Should improve BC before using as warmstart")
        print("  → Try: more data, hyperparameter tuning, or DAgger")
    elif success_rate < 40:
        print("  ⚠️ BC performance is MODERATE (20-40% success)")
        print("  → Can proceed with experiments, but consider improvements")
        print("  → Low performance actually makes research more interesting!")
        print("  → 'Warmstart helps even with imperfect BC' is a strong finding")
    else:
        print("  ✅ BC performance is GOOD (>40% success)")
        print("  → Ready for RL warmstart experiments")
        print("  → This provides solid initialization for RL")
    
    print("="*60 + "\n")
    
    # Generate plots
    plot_diagnostics(results)
    
    return results

if __name__ == "__main__":
    results = evaluate()