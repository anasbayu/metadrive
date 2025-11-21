import multiprocessing as mp
from src.experiment.rl.run_experiments import run_all_experiments


def main():
    run_all_experiments()
    print("Experiments completed.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    main()
