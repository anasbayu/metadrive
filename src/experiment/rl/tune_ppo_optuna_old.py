import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial
from metadrive.envs.metadrive_env import MetaDriveEnv
import torch
import numpy as np
import gymnasium as gym
import os

# --- FOR PyTorch 2.6+ ---
safe_globals = [gym.spaces.box.Box, np.float32, np.int64, np.uint8, np.bool_, np.dtype]
torch.serialization.add_safe_globals(safe_globals)
# ---------------------------

# === CONFIG FOR TUNING ===
N_TRIALS = 35          # How many different hyperparams to try
N_STARTUP_TRIALS = 5   # Don't prune the first 5 trials
N_EVALUATIONS = 5      # How many times to check performance per trial
TIMESTEPS_PER_TRIAL = 6_000_000 # Short runs for speed (6M steps)
NUM_ENV = 5            # Keep your 5 environments
STUDY_NAME = "ppo_metadrive_study_6M" # Name for the database entry
STORAGE_URL = "sqlite:///optuna_study_6M.db" # The file where results are saved
# =========================

# Define the training config (Same as your main script)
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

def create_env(config):
    env = MetaDriveEnv(config)
    return env

def objective(trial):
    """
    Optuna will run this function multiple times with different sampled hyperparameters.
    """
    
    # 1. Sample Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [2048, 3328, 4096])
    
    # Gamma: Higher = better for long tracks. 
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9999])
    
    # GAE Lambda: Trade-off bias vs variance
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 0.99])
    
    # Clip Range: Lower = More stable
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    # N Epochs: Lower = More stable updates
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])

    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: lr={learning_rate}, ent={ent_coef}, batch={batch_size}, n_steps={n_steps}")
    print(f"Gamma={gamma}, GAE={gae_lambda}, Clip={clip_range}, Epochs={n_epochs}")

    # 2. Create Environments
    env_fns = [partial(create_env, TRAIN_CONFIG) for _ in range(NUM_ENV)]
    train_env = SubprocVecEnv(env_fns)
    eval_env = create_env(TRAIN_CONFIG) 

    # 3. Create the Pruning Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=max(TIMESTEPS_PER_TRIAL // N_EVALUATIONS // NUM_ENV, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # 4. Setup the PPO Model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=n_epochs,
        verbose=0, 
        max_grad_norm=0.5 
    )

    # 5. Train with Pruning
    try:
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL, callback=eval_callback)
        mean_reward = eval_callback.last_mean_reward
        
        # Report final value to Optuna
        trial.report(mean_reward, TIMESTEPS_PER_TRIAL)
        
    except optuna.exceptions.TrialPruned:
        train_env.close()
        eval_env.close()
        raise 
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        train_env.close()
        eval_env.close()
        return -1000 

    # 6. Clean up
    train_env.close()
    eval_env.close()
    
    return mean_reward

if __name__ == "__main__":
    
    # ---
    # NEW: Set up persistent storage (SQLite) for the dashboard
    # ---
    print(f"Using database storage: {STORAGE_URL}")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,  # <-- Save to file
        load_if_exists=True,  # <-- Resume if interrupted
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=TIMESTEPS_PER_TRIAL // 5)
    )
    
    print(f"Starting optimization with {N_TRIALS} trials...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    print("\n" + "="*30)
    print("       OPTIMIZATION COMPLETE       ")
    print("="*30)
    
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (Reward): {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to a file
    import json
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Saved to best_hyperparameters.json")