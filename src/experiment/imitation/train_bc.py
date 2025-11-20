import os
import numpy as np
import gymnasium as gym
from functools import partial
import torch

# Import for Imitation Learning (BC)
from imitation.algorithms import bc
import imitation.data.types as im_types
import np_utils

# Imports for SB3 policy and environment
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

# Import for MetaDrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.utils import set_random_seed

# ============== CONFIGURATIONS =============
# ---
# Path to the expert data
# ---
EXPERT_DATA_PATH = "file/expert_data/expert_metadrive_buffer_3.npz"

# ---
# Path to save the new pre-trained IL policy
# ---
BC_POLICY_SAVE_PATH = "file/model/bc_policy_4.zip"

# ---
# We need a base config to create the env for its observation space
# This should match the config used for data collection
# ---
BC_TRAIN_CONFIG = {
    "use_render": False,
    "manual_control": False,
    "log_level": 50,
    "num_scenarios": 100,
    "start_seed": 0,
    "traffic_density": 0.0,
    "window_size": (100, 100),
}

# ---
# Config for evaluating the trained BC policy
# ---
BC_EVAL_CONFIG = {
    "num_scenarios": 10,
    "start_seed": 1000,
    "use_render": False,
    "manual_control": False,
    "log_level": 20,
    "traffic_density": 0.0,
    "window_size": (100, 100),
}
# -------------------------------------------

def create_env_bc(config):
    """
    Simple helper to create a single MetaDrive instance.
    """
    env = MetaDriveEnv(config)
    return env

def evaluate_bc_policy(policy, config, num_episodes=10):
    """
    Evaluates the newly trained BC policy.
    """
    print(f"\n--- Starting Evaluation of BC Policy ---")
    
    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
        
    eval_env = create_env_bc(config)
    set_random_seed(12345) # Use a fixed eval seed

    total_rewards = []
    num_success = 0
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            # Use deterministic=True for evaluation
            action, _states = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")
        total_rewards.append(total_reward)
        if info.get("arrive_dest", False):
            num_success += 1
            print("... Arrived at destination!")
    
    eval_env.close()
    print("--- BC Evaluation Finished ---")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Success Rate: {num_success / num_episodes * 100:.1f}%")


def main():
    """
    Main function to load data, train BC, and save the policy.
    """
    print("--- Starting Behavioral Cloning (BC) Training ---")
    
    # Check if expert data exists
    if not os.path.exists(EXPERT_DATA_PATH):
        print(f"Error: Expert data not found at {EXPERT_DATA_PATH}")
        print("Please run 'collect_expert_data.py' first.")
        return

    # Load the expert data from the .npz file
    print(f"Loading expert data from {EXPERT_DATA_PATH}...")
    data = np.load(EXPERT_DATA_PATH, allow_pickle=True)
    num_transitions = len(data["observations"])
    
    # Check if 'infos' was saved. If not, create dummy infos.
    if "infos" in data:
        infos = data["infos"]
    else:
        print("Warning: 'infos' field not found in expert data. Creating dummy 'infos'.")
        # BC trainer requires an 'infos' field, even if it's empty.
        infos = np.array([{} for _ in range(num_transitions)])

    # Ensure dones is a 1D boolean array
    dones = data["dones"].astype(bool).flatten()

    # Create the 'Transitions' object that imitation's BC trainer expects
    # Note: BC (supervised learning) does not use 'rewards'.
    expert_data = im_types.Transitions(
        obs=data["observations"],
        acts=data["actions"],
        next_obs=data["next_observations"],
        dones=dones,
        infos=infos
    )

    print(f"Loaded {len(expert_data)} expert transitions.")

    # Create a DummyVecEnv for the BC trainer
    # The trainer needs this to get observation/action space info
    venv = DummyVecEnv([partial(create_env_bc, BC_TRAIN_CONFIG)])

    # Set up the BC Trainer
    # We will use the same policy architecture as your PPO agent
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # Use a fixed random seed for reproducible BC training
    rng = np.random.default_rng(0)
    
    print("Initializing BC trainer...")
    policy = ActorCriticPolicy(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        lr_schedule=lambda _: 0.0, # Dummy LR, BC will override this
        **policy_kwargs
    )

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_data,
        policy=policy,
        rng=rng,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 5. Train the policy
    print("--- Starting BC policy training... ---")
    # n_epochs=100 is a good starting point for BC
    bc_trainer.train(n_epochs=10000, log_interval=10)
    print("--- BC Training Complete ---")

    # 6. Save the trained policy
    # bc_trainer.policy is the actual Stable-Baselines3 policy object
    print(f"Saving pre-trained policy to {BC_POLICY_SAVE_PATH}...")
    bc_trainer.policy.save(BC_POLICY_SAVE_PATH)
    print("--- Policy Saved ---")
    
    # 7. Evaluate the new policy
    # evaluate_bc_policy(bc_trainer.policy, BC_EVAL_CONFIG)
    
    venv.close()

if __name__ == "__main__":
    main()