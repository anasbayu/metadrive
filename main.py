import argparse
import multiprocessing as mp
import sys

from src.experiment.rl.run_experiments import run_all_experiments
import src.experiment.rl.optuna_tuning.tune_LeakyPPO_optuna as tune_leakyPPO
import src.experiment.rl.optuna_tuning.tune_PPO_optuna as tune_ppo

def main():
    parser = argparse.ArgumentParser(description="MetaDrive Experiment Runner")
    
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--run-experiments", 
        action="store_true", 
        help="Run the full training and evaluation experiments (PPO/LeakyPPO seeds)."
    )
    
    group.add_argument(
        "--tune-ppo", 
        action="store_true", 
        help="Run Optuna hyperparameter tuning for Standard PPO."
    )
    
    group.add_argument(
        "--tune-leaky-ppo", 
        action="store_true", 
        help="Run Optuna hyperparameter tuning for LeakyPPO."
    )

    group.add_argument(
        "--compare-results", 
        action="store_true", 
        help="Compare the best hyperparameters from PPO and LeakyPPO tuning."
    )

    group.add_argument(
        "--analyze-rliable", 
        action="store_true", 
        help="Run RLiable analysis on the results."
    )

    args = parser.parse_args()

    # 2. Execute the selected function
    if args.run_experiments:
        print(">>> Starting Full Experiments...")
        run_all_experiments()
        
    elif args.tune_ppo:
        print(">>> Starting PPO Hyperparameter Tuning...")
        # Assuming tune_ppo_optuna.py has a main() function or you want to run its logic
        # If it doesn't have main(), you might need to refactor it slightly 
        # or call the optimization function directly.
        if hasattr(tune_ppo, 'main'):
            tune_ppo.main()
        else:
            print("Error: src.experiment.rl.tune_ppo_optuna has no 'main()' function.")
            
    elif args.tune_leaky_ppo:
        print(">>> Starting LeakyPPO Hyperparameter Tuning...")
        if hasattr(tune_leakyPPO, 'main'):
            tune_leakyPPO.main()
        else:
            print("Error: src.experiment.rl.tune_leakyPPO_optuna has no 'main()' function.")

    elif args.compare_results:
        print(">>> Comparing Best Hyperparameters from PPO and LeakyPPO Tuning...")
        import src.experiment.rl.compare_optuna_result as compare_optuna_result
        if hasattr(compare_optuna_result, 'main'):
            compare_optuna_result.main()
        else:
            print("Error: src.experiment.rl.compare_optuna_result has no 'main()' function.")

    elif args.analyze_rliable:
        print(">>> Starting RLiable Analysis...")
        import src.experiment.analyze_with_rliable as analyze_with_rliable
        if hasattr(analyze_with_rliable, 'main'):
            analyze_with_rliable.main()
        else:
            print("Error: src.experiment.analyze_with_rliable has no 'main()' function.")
    print("\n>>> Execution Finished.")

if __name__ == "__main__":
    # Windows multiprocessing fix
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    main()
