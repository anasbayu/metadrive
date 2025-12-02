import os
import json
import numpy as np
import torch
import torch.nn as nn
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from functools import partial

import gymnasium as gym
from stable_baselines3 import PPO
from src.leaky_PPO import LeakyPPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Imitation library for BC warmstart
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Transitions
from stable_baselines3.ppo import MlpPolicy


# ============== LOAD OPTUNA RESULTS ==============
def load_optuna_params(algo_name):
    """Load Optuna-optimized hyperparameters."""
    # Handle suffixes like _Warmstart
    # Check Leaky FIRST because "LeakyPPO" contains "PPO"
    if "LeakyPPO" in algo_name:
        base_algo = "LeakyPPO"
        filename = "leaky_ppo_metadrive_optuna_1.5M_best_params.json"
    elif "PPO" in algo_name:
        base_algo = "PPO"
        filename = "ppo_metadrive_optuna_1.5M_best_params.json"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    if not os.path.exists(filename):
        print(f"Warning: Optuna file {filename} not found. Using defaults.")
        return {}
    
    with open(filename, "r") as f:
        params = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"  LOADED OPTUNA HYPERPARAMETERS FOR {base_algo}")
    print(f"{'='*70}")
    for key, value in params.items():
        print(f"  {key:<20} {value}")
    print(f"{'='*70}\n")
    
    return params


# ============== CONFIGURATIONS =============
NUM_ENV = 5
TIMESTEPS = 15_000_000
PATH_LOG_DIR = "./logs/warmstarted/"
PATH_SAVED_MODEL_ROOT = "./models/warmstarted/"

TRAIN_CONFIG = {
    "num_scenarios": 100,
    "start_seed": 0,
    "use_render": False,
    "manual_control": False,
    "log_level": 50,
    "traffic_density": 0.0,
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "success_reward": 30.0,
    "use_lateral_reward": True
}

EVAL_CONFIG = {
    "num_scenarios": 10,
    "start_seed": 1000,
    "use_render": False,
    "manual_control": False,
    "log_level": 50,
    "traffic_density": 0.0
}

# ============== CALLBACKS =============
class StabilityMetricsCallback(BaseCallback):
    """Tracks stability metrics during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
            
            self.logger.record("stability/reward_mean", np.mean(ep_rewards))
            self.logger.record("stability/reward_std", np.std(ep_rewards))
            self.logger.record("stability/reward_min", np.min(ep_rewards))
            self.logger.record("stability/reward_max", np.max(ep_rewards))
            self.logger.record("stability/ep_len_mean", np.mean(ep_lengths))
            self.logger.record("stability/ep_len_std", np.std(ep_lengths))
            
            success_count = sum([1 for ep in self.model.ep_info_buffer 
                               if ep.get("arrive_dest", False)])
            success_rate = success_count / len(self.model.ep_info_buffer) if len(self.model.ep_info_buffer) > 0 else 0
            self.logger.record("stability/success_rate", success_rate)
        
        return True
    
    def _on_step(self) -> bool:
        return True

# ============== HELPER FUNCTIONS =============
def create_env(config, log_dir=None, seed=0):
    """Create and configure MetaDrive environment."""
    env = MetaDriveEnv(config)
    env.action_space.seed(seed)

    if log_dir:
        monitor_log_dir = os.path.join(log_dir, f"monitor_seed_{seed}_pid_{os.getpid()}")
        os.makedirs(monitor_log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(monitor_log_dir, "monitor.csv"), 
                      info_keywords=("arrive_dest",))
    else:
        env = Monitor(env, log_dir, info_keywords=("arrive_dest",))

    return env

# ============== TRAINING FUNCTION =============
def train(
        algo_name: str,
        experiment_seed: int, 
        experiment_name: str,
        total_timesteps: int = TIMESTEPS,
        leaky_alpha: float = None, 
        bc_model_path: str = None,
        bc_stats_path: str = None,
    ):
    set_random_seed(experiment_seed)

    # Load Optuna hyperparameters
    optuna_params = load_optuna_params(algo_name)
    
    # Extract alpha for LeakyPPO (if present)
    if "LeakyPPO" in algo_name:
        if leaky_alpha is None:
            leaky_alpha = optuna_params.pop("alpha", 0.1) 
        else:
            optuna_params.pop("alpha", None) 

    # Create directories
    run_log_dir = os.path.join(PATH_LOG_DIR, experiment_name)
    path_saved_model = os.path.join(PATH_SAVED_MODEL_ROOT, experiment_name)
    path_checkpoint = os.path.join(path_saved_model, "checkpoints/")
    
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(path_checkpoint, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING: {experiment_name}")
    print(f"  Seed: {experiment_seed}")
    print(f"  Total Timesteps: {total_timesteps:,}")
    if "LeakyPPO" in algo_name:
        print(f"  Algorithm: LeakyPPO (alpha={leaky_alpha})")
    else:
        print(f"  Algorithm: PPO")
    print(f"{'='*70}\n")

    # Create training environment
    train_env = SubprocVecEnv([
        partial(create_env, TRAIN_CONFIG, run_log_dir, seed=experiment_seed + i) 
        for i in range(NUM_ENV)
    ])

    # 2. Warmstart: Apply Normalization
    if bc_stats_path and os.path.exists(bc_stats_path):
        print(f"  [Warmstart] Loading VecNormalize stats from {bc_stats_path}")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        stats = np.load(bc_stats_path)
        train_env.obs_rms.mean = stats["mean"]
        train_env.obs_rms.var = stats["std"] ** 2
        train_env.training = True 
    else:
        # Optional: You can enable VecNormalize even for non-warmstart if desired
        pass
    
    logger = configure(run_log_dir, ["stdout", "csv", "json", "tensorboard"])
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // NUM_ENV,
        save_path=path_checkpoint,
        name_prefix=f"{algo_name.lower()}_checkpoint",
        save_vecnormalize=True
    )
    
    stability_callback = StabilityMetricsCallback()
    
    callbacks = CallbackList([checkpoint_callback, stability_callback])

    # Common parameters
    common_params = {
        "policy": "MlpPolicy",
        "env": train_env,
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
        "tensorboard_log": PATH_LOG_DIR,
        "seed": experiment_seed,
        **optuna_params 
    }

    # Create model (Robust String Check)
    if "LeakyPPO" in algo_name:
        print(f"Instantiating LeakyPPO (alpha={leaky_alpha})...")
        model = LeakyPPO(
            **common_params,
            alpha=leaky_alpha
        )
    elif "PPO" in algo_name:
        print("Instantiating standard PPO...")
        model = PPO(**common_params)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Must be 'PPO' or 'LeakyPPO'.")

    model.set_logger(logger)
    

    # 4. Warmstart: Brain Transplant (Imitation Library Compatible)
    if bc_model_path and os.path.exists(bc_model_path):
        print(f"  [Warmstart] Loading Imitation/SB3 policy from {bc_model_path}...")
        
        try:
            # === FIX START: Handle PyTorch 2.6+ Security Change ===
            import functools
            
            # Create a custom loader that forces weights_only=False
            # This is safe because YOU created the BC model file.
            custom_load = functools.partial(torch.load, weights_only=False)

            # We temporarily swap torch.load with our custom version
            original_load = torch.load
            torch.load = custom_load
            
            try:
                # Now this call uses the modified torch.load internally
                bc_policy = model.policy_class.load(bc_model_path, device="cpu")
            finally:
                # Restore the original torch.load immediately after
                torch.load = original_load
            # === FIX END ===

            # Extract the state dictionary
            bc_weights = bc_policy.state_dict()
            
            # Load these weights into your new PPO model
            model.policy.load_state_dict(bc_weights, strict=False)
            
            print("  [Warmstart] Weights transplanted successfully!")

        except Exception as e:
            print(f"  [Warmstart] CRITICAL ERROR loading BC weights: {e}")
            raise e

    # Train
    print(f"Starting training...")
    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name=experiment_name,
        callback=callbacks,
        reset_num_timesteps=True
    )

    # Save final model
    final_model_path = os.path.join(PATH_SAVED_MODEL_ROOT, f"{experiment_name}_final.zip")
    model.save(final_model_path)

    # Save VecNormalize stats if used
    if isinstance(train_env, VecNormalize):
        stats_path = os.path.join(PATH_SAVED_MODEL_ROOT, f"{experiment_name}_vecnormalize.pkl")
        train_env.save(stats_path)
        print(f"  Saved VecNormalize stats to {stats_path}")

    train_env.close()

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE: {experiment_name}")
    print(f"  Model saved: {final_model_path}")
    print(f"{'='*70}\n")

    return final_model_path


# ============== EVALUATION FUNCTION =============
def evaluate_model(model_path: str, algo_name: str, num_episodes=100):
    """
    Evaluate trained model.
    Returns list of episode rewards.
    """
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {model_path}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return []

    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
        
    # Determine if there is a corresponding VecNormalize file
    vec_norm_path = model_path.replace("_final.zip", "_vecnormalize.pkl")
    use_norm = os.path.exists(vec_norm_path)
    
    if use_norm:
        print(f"  Found VecNormalize stats: {vec_norm_path}")
        # To use VecNormalize, we need a VecEnv (even if it's size 1)
        eval_env = DummyVecEnv([partial(create_env, EVAL_CONFIG)])
        eval_env = VecNormalize.load(vec_norm_path, eval_env)
        eval_env.training = False # Do not update stats during eval
        eval_env.norm_reward = False # Return raw rewards for metric tracking
    else:
        eval_env = MetaDriveEnv(EVAL_CONFIG)


    # Load model (Check Leaky FIRST)
    print(f"Loading {algo_name} model...")
    if "LeakyPPO" in algo_name:
        model = LeakyPPO.load(model_path, env=eval_env)
    elif "PPO" in algo_name:
        model = PPO.load(model_path, env=eval_env)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for ep in range(num_episodes):
        obs = eval_env.reset()
        if not use_norm: obs = obs[0] # Handle non-vec env tuple return

        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)

            if use_norm:
                obs, reward, dones, infos = eval_env.step(action)
                done = dones[0]
                episode_reward += reward[0] 
                info = infos[0]
            else:
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get("arrive_dest", False):
            success_count += 1
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    eval_env.close()
    
    # Summary
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = success_count / num_episodes
    
    # === EXACT LOGGING FORMAT REQUESTED ===
    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Mean Reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min Reward:    {np.min(episode_rewards):.2f}")
    print(f"  Max Reward:    {np.max(episode_rewards):.2f}")
    print(f"  Success Rate:  {success_rate:.1%}")
    print(f"  Mean Length:   {np.mean(episode_lengths):.1f}")
    print(f"{'='*70}\n")
    
    return episode_rewards