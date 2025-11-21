import json
import numpy as np
import os

from src.experiment.rl.train_ppo import train, evaluate_model 

# === EXPERIMENTS CONFIG ===
ALGORITHMS_TO_TEST = ["PPO"]     # ["PPO", "LeakyPPO"]
SEEDS = [0, 5, 10, 15, 20]  # (5 runs per algorithm)
REAL_TIMESTEPS = 15_000_000
EVAL_EPISODES = 50

def run_all_experiments():
    """
    Runs training and evaluation for all algorithms and seeds.
    Stores all results in a JSON file for RLiable analysis.
    """
    # A dictionary to store all scores for RLiable
    all_scores = {}

    print("===== STARTING EXPERIMENT SUITE =====")
    print(f"Total Timesteps: {REAL_TIMESTEPS}")
    print(f"Eval Episodes:   {EVAL_EPISODES}")

    for algo_name in ALGORITHMS_TO_TEST:
        print(f"\n===== STARTING RUNS FOR: {algo_name} =====")
        all_scores[algo_name] = [] # Initialize list for this algo

        for seed in SEEDS:
            # 1. TRAIN
            experiment_name = f"{algo_name}_seed_{seed}"
            print(f"\n--- Training {experiment_name} ---")
            
            model_path = train(
                algo_name=algo_name,
                experiment_seed=seed,
                experiment_name=experiment_name,
                total_timesteps=REAL_TIMESTEPS,
                leaky_alpha=0.01        # Only used if algo_name is 'LeakyPPO'
            )
            
            # 2. EVALUATE
            print(f"\n--- Evaluating {experiment_name} ---")
            
            scores_for_this_seed = evaluate_model(
                model_path=model_path,
                algo_name=algo_name,
                num_episodes=EVAL_EPISODES
            )
            
            # 3. STORE RESULTS
            all_scores[algo_name].append(scores_for_this_seed)

    print("\n===== EXPERIMENT SUITE FINISHED =====")

    # 4. SAVE SCORES FOR RLIABLE
    output_file = "all_experiment_scores.json"
    print(f"Saving all scores to {output_file}...")

    # Convert numpy arrays to lists for JSON serialization
    scores_for_json = {
        key: np.array(value).tolist() for key, value in all_scores.items()
    }

    with open(output_file, 'w') as f:
        json.dump(scores_for_json, f, indent=4)

    print("Done. You can now load 'all_experiment_scores.json' into RLiable.")


if __name__ == "__main__":
    run_all_experiments()