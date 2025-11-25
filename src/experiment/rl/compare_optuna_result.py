import json


def main():
    # Load both results
    with open("ppo_metadrive_optuna_1.5M_best_params.json", "r") as f:
        ppo_params = json.load(f)

    with open("leaky_ppo_metadrive_optuna_1.5M_best_params.json", "r") as f:
        leaky_params = json.load(f)

    # Compare
    print(f"{'Parameter':<20} {'PPO':<20} {'LeakyPPO':<20}")
    print("="*60)
    for key in ppo_params.keys():
        ppo_val = ppo_params.get(key, "N/A")
        leaky_val = leaky_params.get(key, "N/A")
        print(f"{key:<20} {str(ppo_val):<20} {str(leaky_val):<20}")


if __name__ == "__main__":
    main()