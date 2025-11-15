import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO

from train_metadrive_corrected import EVAL_CONFIG, create_env
from imitation.policies.serialize import reconstruct_policy

# ============== CONFIGURATIONS =============

# ---
# ! EVALUATION DICTIONARY !
# Add all models that will be evaluated here.
# - 'path' is the path to the saved model file.
# - 'type' must be 'ppo' for PPO models or 'bc' for BC models.
# ---
MODELS_TO_EVALUATE = {
    "PPO_Run_Seed_0": {
        "type": "ppo",
        "path": "file/model/PPO_7/seed_0/final_model.zip"
    },
    "PPO_Run_Seed_1": {
        "type": "ppo",
        "path": "file/model/PPO_7/seed_1/final_model.zip" 
    },
    "PPO_Run_Seed_2": {
        "type": "ppo",
        "path": "file/model/PPO_7/seed_2/final_model.zip"
    },
    "BC_Agent_200k": {
        "type": "bc",
        "path": "bc_policy.zip"
    },
}

# Number of episodes to run for each agent to get stable stats
NUM_EVAL_EPISODES = 50 

# Use a fixed seed for the evaluation scenarios
EVAL_SEED = 12345
# -------------------------------------------


def run_evaluation():
    """
    Loads and evaluates all models in the MODELS_TO_EVALUATE dict.
    """
    print(f"--- Starting Evaluation ---")
    print(f"Running {NUM_EVAL_EPISODES} episodes for each of {len(MODELS_TO_EVALUATE)} agents.")
    
    all_results = []

    # Use a single, non-vectorized env for evaluation
    # We create it *once* to be fair
    eval_env = create_env(EVAL_CONFIG, seed=EVAL_SEED)

    for model_name, model_info in MODELS_TO_EVALUATE.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        
        # Loading the model
        model_path = model_info["path"]
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
            
        try:
            if model_info["type"] == "ppo":
                model = PPO.load(model_path)
            elif model_info["type"] == "bc":
                model = reconstruct_policy(model_path)
            else:
                print(f"Unknown model type '{model_info['type']}'. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading model {model_name}: {e}. Skipping.")
            continue

        # Running N episodes of evaluation
        for i in range(NUM_EVAL_EPISODES):
            if (i + 1) % 10 == 0:
                print(f"  ... running episode {i + 1}/{NUM_EVAL_EPISODES}")
                
            obs, info = eval_env.reset()
            
            terminated = False
            truncated = False
            
            total_reward = 0.0
            total_cost = 0.0
            total_steps = 0
            
            while not terminated and not truncated:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                # Collect metrics at each step
                total_reward += reward
                total_cost += info.get("cost", 0.0) # Cost (like out_of_road)
                total_steps += 1
            
            # Collect metrics at the end of the episode
            success = info.get("arrive_dest", False)
            crash = (
                info.get("crash_vehicle", False) or 
                info.get("crash_object", False) or 
                info.get("crash_sidewalk", False)
            )
            
            # Store results for this single episode
            all_results.append({
                "model_name": model_name,
                "reward": total_reward,
                "success": 1.0 if success else 0.0,
                "crash": 1.0 if crash else 0.0,
                "cost": total_cost,
                "ep_length": total_steps
            })

    eval_env.close()
    
    if not all_results:
        print("No models were evaluated. Exiting.")
        return

    # Process and print the final results
    print("\n\n--- Evaluation Complete. Final Results ---")
    
    # Convert list of dicts to a Pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Group by model name and calculate mean/std for all metrics
    grouped = results_df.groupby("model_name")
    
    mean_stats = grouped.mean()
    std_stats = grouped.std()
    
    # Format the final table
    final_table = pd.DataFrame()
    for col in mean_stats.columns:
        final_table[f"{col}_mean"] = mean_stats[col]
        final_table[f"{col}_std"] = std_stats[col]
        
    # Special formatting for rates
    final_table["success_mean"] = (final_table["success_mean"] * 100).map('{:.1f}%'.format)
    final_table["crash_mean"] = (final_table["crash_mean"] * 100).map('{:.1f}%'.format)
    
    print(final_table.to_string())

if __name__ == "__main__":
    run_evaluation()