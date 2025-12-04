import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
# from stable_baselines3 import PPO
from src.leaky_PPO import LeakyPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from functools import partial
from metadrive.envs.metadrive_env import MetaDriveEnv
import torch
import numpy as np
import gymnasium as gym
import os
import json
import gc

# --- FOR PyTorch 2.6+ ---
safe_globals = [gym.spaces.box.Box, np.float32, np.int64, np.uint8, np.bool_, np.dtype]
torch.serialization.add_safe_globals(safe_globals)
# ---------------------------

# === CONFIG FOR TUNING ===
N_TRIALS = 100           # Total number of trials to run
N_STARTUP_TRIALS = 5     # Number of trials before pruning starts
N_EVALUATIONS = 15       # Evaluate 15 times during training (to decide pruning)
TIMESTEPS_PER_TRIAL = 1_500_000  # 1.5M steps per trial
NUM_ENV = 50
EVAL_EPISODES = 5       # Episodes per evaluation
STUDY_NAME = "leaky_ppo_optuna_1.5M_new"
STORAGE_URL = "sqlite:///optuna_leaky_ppo_1.5M_new.db"
TRAFFIC_DENSITY = 0.3    # Traffic density in MetaDrive (Must match between training and evaluation)
# =========================

# Training config (match your main script)
TRAIN_CONFIG = {
    "num_scenarios": 100,
    "start_seed": 0,
    "use_render": False,
    "manual_control": False,
    "log_level": 50,
    "traffic_density": TRAFFIC_DENSITY,
    "out_of_road_penalty": 30.0,
    "crash_vehicle_penalty": 30.0,
    "crash_object_penalty": 30.0,
    "success_reward": 100.0,
    "use_lateral_reward": True              # Keeps it centered
}

# Evaluation config (simpler for faster tuning)
EVAL_CONFIG = {
    "num_scenarios": 10,
    "start_seed": 1000,
    "use_render": False,
    "manual_control": False,
    "log_level": 50,
    "traffic_density": TRAFFIC_DENSITY,
}


def create_env(config, seed=0):
    """Create and seed MetaDrive environment."""
    env_config = config.copy()
    env_config["start_seed"] = seed     # Force the MetaDrive internal seed to be unique per process

    env = MetaDriveEnv(env_config)
    env = Monitor(env)
    env.action_space.seed(seed)
    set_random_seed(seed)
    return env


class TrialEvalCallback(BaseCallback):
    """
    Callback for evaluating and reporting to Optuna during training.
    This enables pruning of unpromising trials.
    """
    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run evaluation episodes
            episode_rewards = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                terminated = False
                truncated = False
                
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            
            # Report to Optuna
            self.trial.report(mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            if self.verbose > 0:
                print(f"  Eval {self.eval_idx}: Mean Reward = {mean_reward:.2f}")
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Stop training
        
        return True


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    Returns mean reward on evaluation episodes.
    """
    
    # ========================================
    # SAMPLE HYPERPARAMETERS
    # ========================================
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    n_epochs = trial.suggest_int("n_epochs", 5, 10)
    target_kl = trial.suggest_float("target_kl", 0.01, 0.05)
    # --- NETWORK ARCHITECTURE ---
    net_arch_type = trial.suggest_categorical("net_arch", ["wide", "medium", "deep"])
    net_arch_map = {
        "wide":  dict(pi=[400, 300], vf=[400, 300]),        # Wide architecture from DDPG
        "medium": dict(pi=[256, 256], vf=[256, 256]),
        "deep":   dict(pi=[256, 256, 256], vf=[256, 256, 256])
    }
    selected_arch = net_arch_map[net_arch_type]
    # ----------------------------
    alpha = trial.suggest_float("alpha", 0.01, 0.2)
    
    trial_seed = trial.number
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | Seed: {trial_seed}")
    print(f"{'='*60}")
    print(f"  learning_rate:  {learning_rate:.6f}")
    print(f"  ent_coef:       {ent_coef:.6f}")
    print(f"  batch_size:     {batch_size}")
    print(f"  n_steps:        {n_steps}")
    print(f"  gamma:          {gamma:.4f}")
    print(f"  gae_lambda:     {0.95:.4f}")
    print(f"  max_grad_norm:  {0.5:.2f}")
    print(f"  clip_range:     {clip_range:.2f}")
    print(f"  n_epochs:       {n_epochs}")
    print(f"  target_kl:      {target_kl:.4f}")
    print(f"  net_arch:       {net_arch_type} -> {selected_arch}")
    print(f"  alpha:          {alpha:.4f}")
    print(f"{'='*60}\n")
    
    # ========================================
    # CREATE ENVIRONMENTS
    # ========================================
    try:
        # Training environments (vectorized)
        train_env = SubprocVecEnv([
            partial(create_env, TRAIN_CONFIG, seed=trial_seed + i) 
            for i in range(NUM_ENV)
        ])
        
        # Evaluation environment (single)
        eval_env = create_env(EVAL_CONFIG, seed=trial_seed + 1000)
        
    except Exception as e:
        print(f"Failed to create environments: {e}")
        return -1000.0
    
    # ========================================
    # CREATE PPO MODEL
    # ========================================
    policy_kwargs = dict(
        net_arch=selected_arch,
    )
    
    try:
        model = LeakyPPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=0.95,        # Fixed to 0.95 as per industry standard
            max_grad_norm=0.5,      # Fixed to 0.5 as per industry standard
            clip_range=clip_range,
            n_epochs=n_epochs,
            target_kl=target_kl,
            alpha=alpha,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=trial_seed
        )
    except Exception as e:
        print(f"Failed to create model: {e}")
        train_env.close()
        eval_env.close()
        return -1000.0
    
    # ========================================
    # CREATE EVALUATION CALLBACK
    # ========================================
    eval_freq = max(TIMESTEPS_PER_TRIAL // N_EVALUATIONS // NUM_ENV, 1000)
    
    eval_callback = TrialEvalCallback(
        eval_env=eval_env,
        trial=trial,
        n_eval_episodes=EVAL_EPISODES,
        eval_freq=eval_freq,
        deterministic=True,
        verbose=1
    )
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    try:
        model.learn(
            total_timesteps=TIMESTEPS_PER_TRIAL,
            callback=eval_callback,
            reset_num_timesteps=True
        )
        
        # Check if pruned
        if eval_callback.is_pruned:
            print(f"Trial {trial.number} was pruned.")
            raise optuna.exceptions.TrialPruned()
        
        # Final evaluation
        print(f"\nFinal evaluation for Trial {trial.number}...")
        final_episode_rewards = []
        
        for ep in range(EVAL_EPISODES):
            obs, _ = eval_env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
            
            final_episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(final_episode_rewards)
        std_reward = np.std(final_episode_rewards)
        
        print(f"Trial {trial.number} Final Result: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except optuna.exceptions.TrialPruned:
        train_env.close()
        eval_env.close()
        raise
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        train_env.close()
        eval_env.close()
        return -1000.0
    
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        
        gc.collect()
    
    # ========================================
    # CLEANUP
    # ========================================
    if train_env is not None:
        train_env.close()
    if eval_env is not None:
        eval_env.close()
    
    return mean_reward


def run_optimization():
    """Run Optuna optimization study."""
    
    print(f"\n{'='*70}")
    print(f"  STARTING HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Study Name:     {STUDY_NAME}")
    print(f"Storage:        {STORAGE_URL}")
    print(f"Total Trials:   {N_TRIALS}")
    print(f"Steps/Trial:    {TIMESTEPS_PER_TRIAL:,}")
    print(f"Environments:   {NUM_ENV}")
    print(f"{'='*70}\n")
    
    # Create study with pruning
    sampler = TPESampler(seed=42)  # Reproducible sampling
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=N_EVALUATIONS // 2  # Don't prune too early
    )
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,  # Resume if interrupted
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
            catch=(Exception,)  # Continue even if some trials fail
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user.")
    
    # ========================================
    # RESULTS
    # ========================================
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\nCompleted Trials: {len(study.trials)}")
    print(f"Pruned Trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed Trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if study.best_trial:
        print(f"\n{'='*70}")
        print(f"  BEST TRIAL")
        print(f"{'='*70}")
        print(f"Trial Number: {study.best_trial.number}")
        print(f"Best Reward:  {study.best_value:.2f}")
        print(f"\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key:<20} {value}")
        
        # Save to JSON
        output_file = f"{STUDY_NAME}_best_params.json"
        with open(output_file, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print(f"\n✅ Saved to '{output_file}'")
        
        # Save full study info
        study_info = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        }
        
        info_file = f"{STUDY_NAME}_study_info.json"
        with open(info_file, "w") as f:
            json.dump(study_info, f, indent=4)
        print(f"✅ Saved study info to '{info_file}'")
    
    print(f"\n{'='*70}\n")
    
    return study


def main():
    study = run_optimization()

    print("\nTop 5 Trials:")
    print(f"{'Trial':<8} {'Reward':<12} {'State':<12}")
    print("-" * 32)
    
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True
    )[:5]
    
    for trial in sorted_trials:
        print(f"{trial.number:<8} {trial.value:<12.2f} {trial.state.name:<12}")
    
    return study

if __name__ == "__main__":
    study = main()

    # ==== VISUALIZATION ====
    print("\nGenerating analysis plots...")
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Plot 1: Optimization History
        # Shows if the agent is actually getting better over the trials.
        fig1 = plot_optimization_history(study)
        fig1.write_image("leaky_optuna_history.png")
        print(" -> Saved 'leaky_optuna_history.png'")
        
        # Plot 2: Hyperparameter Importance
        # Shows which parameter mattered the most (e.g., "Learning Rate was 50% of the reason we succeeded").
        fig2 = plot_param_importances(study)
        fig2.write_image("leaky_optuna_importance.png")
        print(" -> Saved 'leaky_optuna_importance.png'")
        
    except ImportError:
        print("\n⚠️  Could not generate plots. Missing dependencies.")
        print("   Run: pip install plotly kaleido pandas")
    except Exception as e:
        print(f"\n⚠️  Plotting failed: {e}")