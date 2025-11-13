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

# ============== LOG and Monitoring =============
# WandB integration
# import wandb
# from wandb.integration.sb3 import WandbCallback

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
EXPERIMENT_NAME = "PPO_TEST"
EXPERIMENT_SEED = 0
PATH_SAVED_MODEL = "file/model/" + EXPERIMENT_NAME
PATH_CHECKPOINT = os.path.join(PATH_SAVED_MODEL, "checkpoints/")

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

WANDB_CONFIG = {
    "policy_type": "MlpPolicy",
    "total_timesteps": TIMESTEPS,
    "env_name": "MetaDrive",
    "n_steps": N_STEPS,
    "net_arch": "pi=[256, 256], vf=[256, 256]",
    "num_envs": NUM_ENV,
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
def train(model_path_to_load=None):
    """
    Train a new model from scratch using a specific seed
    and scheduled hyperparameters.
    """
    experiment_name = EXPERIMENT_NAME
    
    """
    experiment seed:
    experiment 1: seed = 0 (PPO_7) 
    experiment 2: seed = 5 (PPO_8)
    experiment 3: seed = 10 (PPO_9)
    experiment 4: seed = 15 (PPO_10)
    experiment 5: seed = 20 (PPO_11)
    """
    seed = EXPERIMENT_SEED
    set_random_seed(seed)

    MODEL_PARAMS = {
        "learning_rate": linear_schedule(1e-4, 1e-6),
        "ent_coef": 0.01
    }

    # Create log directories
    run_log_dir = os.path.join(PATH_LOG_DIR, experiment_name)
    os.makedirs(run_log_dir, exist_ok=True)

    os.makedirs(PATH_CHECKPOINT, exist_ok=True)
    print(f"Starting training for run: {experiment_name}")
    print(f"Logs will be saved to: {PATH_LOG_DIR + experiment_name}")
    print(f"Checkpoints will be saved to: {PATH_CHECKPOINT}")

    train_env = SubprocVecEnv([partial(create_env, TRAIN_CONFIG, run_log_dir) for _ in range(NUM_ENV)])
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

    model = PPO(
            "MlpPolicy", 
            train_env,
            n_steps=N_STEPS,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=PATH_LOG_DIR,
            max_grad_norm=0.5,
            seed=seed,
            **MODEL_PARAMS
    )
    # model = LeakyPPO(
    #         "MlpPolicy", 
    #         train_env,
    #         n_steps=N_STEPS,
    #         verbose=1,
    #         policy_kwargs=policy_kwargs,
    #         tensorboard_log=PATH_LOG_DIR,
    #         max_grad_norm=0.5,
    #         seed=seed,
    #         **MODEL_PARAMS,
    #         alpha=0.01
    # )

    model.set_logger(logger)
    model.learn(
        total_timesteps=TIMESTEPS, 
        tb_log_name=experiment_name,
        callback=checkpoint_callback,
        reset_num_timesteps=True
    )
    model.save(PATH_SAVED_MODEL)

    print("--- Training Complete ---")
    print(f"Model saved to {PATH_SAVED_MODEL}.zip")
    print(f"Logs saved to {run_log_dir}")

def evaluate_model(model_path):
    print("\n--- Starting Evaluation ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
        
    eval_env = MetaDriveEnv(EVAL_CONFIG)

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    num_episodes = 10
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
    
    eval_env.close()
    print("--- Evaluation Finished ---")


def random_policy():
    # function to run a random policy in the environment.

    env = create_env(CONFIG)

    obs, info = env.reset()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()