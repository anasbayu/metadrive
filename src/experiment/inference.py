import os
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from stable_baselines3.common.policies import ActorCriticPolicy
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.utils import set_random_seed

# ============== CONFIGURATION =============
safe_globals = [
    gym.spaces.box.Box,
    np.float32,  # <-- The one from your error message
    np.int64,    # A common type for timesteps/seeds
    np.uint8,    # A common type for image observations
    np.bool_,    # A common type for 'dones'
    np.dtype     # Keep the generic one just in case
]
torch.serialization.add_safe_globals(safe_globals)


# This config is for inference, so we MUST use rendering
# It's based on your EVAL_CONFIG from other scripts
INFERENCE_CONFIG = {
    "num_scenarios": 100, # Use a large pool of scenarios
    "start_seed": 1000,
    "use_render": True,      # <-- MUST BE TRUE to watch
    "manual_control": False,
    "log_level": 20,
    "traffic_density": 0.1,  # You can change this to 0.0 or 0.3
}

# ============== HELPER FUNCTIONS =============
def create_env_inference(config, seed=0):
    """
    Creates a single, non-monitored env for inference.
    """
    env = MetaDriveEnv(config)
    
    # Set the environment's action_space seed to ensure reproducibility
    env.action_space.seed(seed)
    return env

def load_agent(model_path, model_type, venv):
    """
    Loads a PPO or BC agent from a saved file.
    
    BUG FIX: This function now requires the 'venv' (environment) object.
    """
    print(f"Loading {model_type.upper()} model from: {model_path}")

    orig_torch_load = torch.load
    def _torch_load_override(f, *args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_torch_load(f, *args, **kwargs)

    try:
        torch.load = _torch_load_override
        if model_type == "ppo":
            return PPO.load(model_path, env=venv)
        elif model_type == "bc":
            return ActorCriticPolicy.load(model_path, device="cpu")
        else:
            raise ValueError(f"Unknown model type '{model_type}'. Must be 'ppo' or 'bc'.")
    finally:
        # restore original loader to avoid side-effects
        torch.load = orig_torch_load

def run_inference(agent, env, num_episodes=10):
    """
    Runs the main inference loop.
    """
    print(f"\n--- Starting Inference ---")
    print(f"Running {num_episodes} episodes.")
    
    try:
        MetaDriveEnv.setup_assets()
    except Exception:
        pass
    
    for ep in range(num_episodes):
        print(f"--- Episode {ep + 1}/{num_episodes} ---")
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        total_steps = 0
        
        while not terminated and not truncated:
            # Use deterministic=True to get the *best* action (no exploration)
            action, _states = agent.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            
            # The env.render() call is handled by use_render=True
            # in the MetaDrive config
        
        # --- Episode Finished ---
        print(f"  Total Steps: {total_steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        if info.get("arrive_dest", False):
            print("  Result: Arrived at destination!")
        elif info.get("crash_vehicle", False):
            print("  Result: Crashed into a vehicle.")
        elif info.get("crash_object", False) or info.get("crash_sidewalk", False):
            print("  Result: Crashed into an object or curb.")
        else:
            print("  Result: Truncated (timeout).")
            
    env.close()
    print("\n--- Inference Complete ---")

# ============== MAIN EXECUTION =============
def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained MetaDrive agent.")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model .zip file."
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["ppo", "bc"],
        help="The type of model to load ('ppo' or 'bc')."
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run."
    )
    
    args = parser.parse_args()

    # ---
    # BUG FIX: We must create the environment FIRST,
    # because load_agent() now needs it.
    # ---

    # 1. Create the environment
    # We use a fixed seed for inference to get reproducible runs
    set_random_seed(EVAL_SEED)
    env = create_env_inference(INFERENCE_CONFIG, seed=EVAL_SEED)
    
    # 2. Load the agent
    try:
        # Pass the newly created 'env' to the loader
        agent = load_agent(args.model_path, args.model_type, venv=env) 
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close() # Close the env if loading fails
        return

    # 3. Run the inference loop
    run_inference(agent, env, args.episodes)


if __name__ == "__main__":
    # Define a fixed eval seed
    EVAL_SEED = 42
    main()