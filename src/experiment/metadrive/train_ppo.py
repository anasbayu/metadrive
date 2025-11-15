import os
from random import seed
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from functools import partial
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from src.leaky_PPO import LeakyPPO

# default sb3 logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


# ============== Evaluation =============
# import rliable

# ============== CONFIGURATIONS =============
N_STEPS = 3328      # max steps in 1 episode (make sure N_STEPS * NUM_ENV < TIMESTEPS)
NUM_ENV = 5
TIMESTEPS = 15_000 # max total steps the agents take during training
PATH_LOG_DIR = "./logs/"
PATH_SAVED_MODEL_ROOT = "file/model/"

# PATH_SAVED_MODEL = "file/model/" + EXPERIMENT_NAME
# PATH_CHECKPOINT = os.path.join(PATH_SAVED_MODEL, "checkpoints/")
# EXPERIMENT_NAME = "PPO_TEST"
# EXPERIMENT_SEED = 0

TRAIN_CONFIG = {
        "num_scenarios": 100,
        "start_seed": 0,
        "use_render": False,
        "manual_control": False,
        "log_level": 50,
        "traffic_density": 0.0,
        "out_of_road_penalty": 10.0,     # Default is 5.0
        "crash_vehicle_penalty": 10.0,  # Default is 1.0
        "crash_object_penalty": 10.0,   # Default is 1.0
        "use_lateral_reward": True     # Default is False, enables lane-keeping reward
    }

EVAL_CONFIG = {
        "num_scenarios": 10,
        "start_seed": 1000,
        "use_render": True,
        "manual_control": False,
        "log_level": 20,
        "traffic_density": 0.0
    }

# ============== HELPER FUNCTIONS =============
def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
    """
    Linear interpolation between initial_value and end_value.
    
    :param initial_value: The initial value
    :param end_value: The end value
    :return: A function that takes progress_remaining (0.0 to 1.0) 
             and returns the interpolated value.
    """
    def func(progress_remaining: float) -> float:
        # Progress remaining = 1.0 at the beginning, 0.0 at the end
        return (initial_value - end_value) * progress_remaining + end_value

    return func


def create_env(config, log_dir=None, seed=0):
    """
    Helper function to create, configure, wrap, and seed the environment.
    """
    env = MetaDriveEnv(config)
    env.action_space.seed(seed)

    if log_dir:
        monitor_log_dir = os.path.join(log_dir, f"monitor_seed_{seed}_pid_{os.getpid()}")
        os.makedirs(monitor_log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(monitor_log_dir, "monitor.csv"), info_keywords=("arrive_dest",))
    else:
        env = Monitor(env, log_dir, info_keywords=("arrive_dest",))

    return env


# ============== MAIN FUNCTIONS =============
def train(
    algo_name: str,
    experiment_seed: int, 
    experiment_name: str,
    total_timesteps: int = TIMESTEPS,
    leaky_alpha: float = 0.01):
    """
    Train a new model from scratch using a specific seed
    and scheduled hyperparameters.
    """
    
    """
    experiment seed:
    experiment 1: seed = 0 (PPO_7) 
    experiment 2: seed = 5 (PPO_8)
    experiment 3: seed = 10 (PPO_9)
    experiment 4: seed = 15 (PPO_10)
    experiment 5: seed = 20 (PPO_11)
    """
    # seed = EXPERIMENT_SEED
    set_random_seed(experiment_seed)

    MODEL_PARAMS = {
        "learning_rate": linear_schedule(1e-4, 1e-6),
        "ent_coef": 0.01
    }

    # Create log directories
    run_log_dir = os.path.join(PATH_LOG_DIR, experiment_name)
    path_saved_model = os.path.join(PATH_SAVED_MODEL_ROOT, experiment_name)
    path_checkpoint = os.path.join(path_saved_model, "checkpoints/")
    
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(path_checkpoint, exist_ok=True)
    
    print(f"--- Starting training for run: {experiment_name} (Seed: {experiment_seed}) ---")

    train_env = SubprocVecEnv([partial(create_env, TRAIN_CONFIG, run_log_dir, seed=experiment_seed + i) for i in range(NUM_ENV)])
    logger = configure(run_log_dir, ["stdout", "csv", "json", "tensorboard"])
    policy_kwargs = dict(
        # pi = policy network, vf = value function network
        # default is [64, 64], use larger networks for more complex tasks like MetaDrive
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    checkpoint_callback = CheckpointCallback(
      save_freq=499_200, # save checkpoint each 30 updates (timesteps*n_env)
      save_path=PATH_CHECKPOINT,
      name_prefix="ppo_metadrive_ckpt",
      save_replay_buffer=False,
      save_vecnormalize=False
    )

    model = None
    common_params = {
        "policy": "MlpPolicy",
        "env": train_env,
        "n_steps": N_STEPS,
        "verbose": 0,
        "policy_kwargs": policy_kwargs,
        "tensorboard_log": PATH_LOG_DIR,
        "max_grad_norm": 0.5,
        "seed": experiment_seed,
        **MODEL_PARAMS
    }

    if algo_name == "PPO":
        print("Instantiating standard PPO...")
        model = PPO(**common_params)
        
    elif algo_name == "LeakyPPO":
        print(f"Instantiating LeakyPPO (alpha={leaky_alpha})...")
        model = LeakyPPO(
            **common_params,
            alpha=leaky_alpha
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Must be 'PPO' or 'LeakyPPO'.")
    # model = PPO(
    #         "MlpPolicy", 
    #         train_env,
    #         n_steps=N_STEPS,
    #         verbose=1,
    #         policy_kwargs=policy_kwargs,
    #         tensorboard_log=PATH_LOG_DIR,
    #         max_grad_norm=0.5,
    #         seed=experiment_seed,
    #         **MODEL_PARAMS
    # )
    # model = LeakyPPO(
    #         "MlpPolicy", 
    #         train_env,
    #         n_steps=N_STEPS,
    #         verbose=1,
    #         policy_kwargs=policy_kwargs,
    #         tensorboard_log=PATH_LOG_DIR,
    #         max_grad_norm=0.5,
    #         seed=experiment_seed,
    #         **MODEL_PARAMS,
    #         alpha=0.01
    # )

    model.set_logger(logger)
    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name=experiment_name,
        callback=checkpoint_callback,
        reset_num_timesteps=True
    )

    # Save the final model with a unique name
    final_model_path = os.path.join(PATH_SAVED_MODEL_ROOT, f"{experiment_name}_final.zip")
    model.save(final_model_path)

    print(f"--- Training Complete for {experiment_name} ---")

    # Return the path to the saved model for evaluation
    return final_model_path

def evaluate_model(model_path: str, algo_name: str, num_episodes=10):
    print(f"--- Starting Evaluation for {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return [] # Return an empty list on failure

    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
        
    eval_env = MetaDriveEnv(EVAL_CONFIG)

    # Load the trained model
    model = None
    print(f"Loading {algo_name} model...")
    if algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "LeakyPPO":
        # Pass eval_env so SB3 can set it on the loaded model
        model = LeakyPPO.load(model_path, env=eval_env)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    episode_rewards = []
    print(f"Running {num_episodes} evaluation episodes...")
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            # Use deterministic=True for evaluation (no random exploration)
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward            
        
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")
        episode_rewards.append(total_reward) # Store the reward
    
    eval_env.close()
    print("--- Evaluation Finished ---")

    return episode_rewards


def random_policy():
    # function to run a random policy in the environment.

    env = create_env(CONFIG)

    obs, info = env.reset()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()