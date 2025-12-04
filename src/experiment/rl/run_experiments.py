import json
import numpy as np
import os

from src.experiment.rl.train_ppo import train, evaluate_model 

# ================= BC WARMSTART CONFIGURATION =================
# BC_MODEL_PATH = "./file/model/bc_256.zip"
BC_MODEL_PATH = None
BC_STATS_PATH = None

# === EXPERIMENTS CONFIG ===
ALGORITHMS_TO_TEST = ["LeakyPPO"]     # ["PPO", "LeakyPPO", "PPO_Warmstart", "LeakyPPO_Warmstart"]
SEEDS = [0]  # 5 seeds for statistical significance (per algorithm)
TIMESTEPS  = 15_000_000
EVAL_EPISODES = 100 # RLiable recommends at least 100 episodes for evaluation

def run_all_experiments():
    """
    Runs training and evaluation for all algorithms and seeds.
    Stores all results in a JSON file for RLiable analysis.
    """
    all_scores = {}     # A dictionary to store all scores for RLiable

    print("\n" + "="*70)
    print("  EXPERIMENT SUITE - OPTUNA-OPTIMIZED TRAINING")
    print("="*70)
    print(f"  Algorithms:    {', '.join(ALGORITHMS_TO_TEST)}")
    print(f"  Seeds:         {SEEDS}")
    print(f"  Timesteps:     {TIMESTEPS:,}")
    print(f"  Eval Episodes: {EVAL_EPISODES}")
    print("="*70 + "\n")

    for algo_name in ALGORITHMS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"  ALGORITHM: {algo_name}")
        print(f"{'='*70}\n")

        all_scores[algo_name] = [] # Initialize list for this algo

        for seed in SEEDS:
            # Setup paths only if it's the warmstart variant
            bc_model = BC_MODEL_PATH if algo_name == "PPO_Warmstart" or algo_name == "LeakyPPO_Warmstart" else None
            bc_stats = BC_STATS_PATH if algo_name == "PPO_Warmstart" or algo_name == "LeakyPPO_Warmstart" else None


            # 1. TRAIN
            experiment_name = f"{algo_name}_optuna_seed_{seed}_warmstart" if bc_model else f"{algo_name}_optuna_seed_{seed}"
            print(f"\n>>> Training {experiment_name}")
            
            model_path = train(
                algo_name=algo_name,
                experiment_seed=seed,
                experiment_name=experiment_name,
                total_timesteps=TIMESTEPS,
                leaky_alpha=None,        # Will be loaded from Optuna results
                bc_model_path=bc_model,
                bc_stats_path=bc_stats,
            )
            
            # 2. EVALUATE
            print(f"\n>>> Evaluating {experiment_name}")
            
            scores_for_this_seed = evaluate_model(
                model_path=model_path,
                algo_name=algo_name,
                num_episodes=EVAL_EPISODES
            )

            # 3. STORE RESULTS
            all_scores[algo_name].append(scores_for_this_seed)
            

    print("\n" + "="*70)
    print("  EXPERIMENT SUITE COMPLETE")
    print("="*70 + "\n")

    # 4. SAVE SCORES FOR RLIABLE
    output_file = "optuna_experiment_results_warmstart.json"
    print(f"Saving all scores to {output_file}...")

    # Convert to serializable format
    scores_for_json = {
        key: [[float(score) for score in seed_scores] for seed_scores in value]
        for key, value in all_scores.items()
    }

    with open(output_file, 'w') as f:
        json.dump(scores_for_json, f, indent=4)


    # Print summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    for algo_name, scores in all_scores.items():
        flat_scores = np.array(scores).flatten()
        print(f"\n{algo_name}:")
        print(f"  Total Episodes: {len(flat_scores)}")
        print(f"  Mean Reward:    {flat_scores.mean():.2f} ± {flat_scores.std():.2f}")
        print(f"  Median Reward:  {np.median(flat_scores):.2f}")
    print("="*70 + "\n")
    
    print(f"✅ Results saved to '{output_file}'")
    print("   You can now analyze with RLiable!")

if __name__ == "__main__":
    run_all_experiments()