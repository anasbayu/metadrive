import json
import numpy as np
import os
import sys

try:
    # Import the PPO/LeakyPPO evaluator
    from src.experiment.metadrive.train_ppo import evaluate_model
except ImportError:
    print("ERROR: Could not find 'evaluate_model' from 'train_ppo.py'", file=sys.stderr)
    sys.exit(1)

try:
    # Import the BC evaluator
    from src.experiment.imitation.train_bc import evaluate_bc_model
except ImportError:
    print("ERROR: Could not find 'evaluate_bc_model' from 'train_bc.py'", file=sys.stderr)
    sys.exit(1)
# ---------------------------------------

# --- CONFIGURATION ---
EXPERIMENTS_TO_RUN = {
    
    "PPO": [
        "file/model/PPO_7/PPO_7_final.zip",
        "file/model/PPO_8/PPO_8_final.zip",
        "file/model/PPO_9/PPO_9_final.zip",
        "file/model/PPO_10/PPO_10_final.zip",
        "file/model/PPO_11/PPO_11_final.zip"
    ],
    
    "LeakyPPO": [
        "path/to/your/leaky_model_seed_1.zip",
        "path/to/your/leaky_model_seed_2.zip",
        "path/to/your/leaky_model_seed_3.zip",
        "path/to/your/leaky_model_seed_4.zip",
        "path/to/your/leaky_model_seed_5.zip",
    ],
    
    "BC": [
        "path/to/your/bc_model_seed_1.zip",
        "path/to/your/bc_model_seed_2.zip",
        "path/to/your/bc_model_seed_3.zip",
        "path/to/your/bc_model_seed_4.zip",
        "path/to/your/bc_model_seed_5.zip",
    ]
}

# How many episodes to run for each model
EVAL_EPISODES = 50 
# ----------------------

def main():
    """
    Main function to run all evaluations.
    """
    print("===== STARTING EVALUATION OF ALL TRAINED MODELS =====")
    
    all_scores = {}

    # Loop through each algorithm
    for algo_name, model_paths in EXPERIMENTS_TO_RUN.items():
        
        print(f"\n===== Evaluating Algorithm: {algo_name} =====")
        all_scores[algo_name] = []
        
        # Check if all files exist
        valid_paths = True
        for p in model_paths:
            if not os.path.exists(p):
                print(f"!!! ERROR: File not found: {p}", file=sys.stderr)
                valid_paths = False
        
        if not valid_paths:
            print(f"!!! SKIPPING {algo_name} due to missing model files. !!!")
            continue

        # Loop through each model path for this algorithm
        for model_path in model_paths:
            
            print(f"\n--- Evaluating Model: {model_path} ---")
            scores_for_this_model = []
            
            # --- Call the correct evaluation function ---
            try:
                if algo_name in ["PPO", "LeakyPPO"]:
                    scores_for_this_model = evaluate_model(
                        model_path=model_path,
                        algo_name=algo_name,
                        num_episodes=EVAL_EPISODES
                    )
                elif algo_name == "BC":
                    scores_for_this_model = evaluate_bc_model(
                        model_path=model_path,
                        num_episodes=EVAL_EPISODES
                    )
                else:
                    print(f"Warning: No evaluator found for '{algo_name}'. Skipping.", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"!!! ERROR during evaluation of {model_path}: {e}", file=sys.stderr)
                print("Skipping this model and continuing...")
                continue
            # --------------------------------------------
                
            all_scores[algo_name].append(scores_for_this_model)

    print("\n===== EVALUATION SUITE FINISHED =====")

    # Save the final scores to a JSON file
    output_file = "all_algorithms_scores.json"
    print(f"Saving all scores to {output_file}...")

    # Convert to list for JSON
    scores_for_json = {
        key: np.array(value).tolist() for key, value in all_scores.items() if value
    }

    with open(output_file, 'w') as f:
        json.dump(scores_for_json, f, indent=4)

    print(f"Done. You can now load '{output_file}' into RLiable.")


if __name__ == "__main__":
    main()