import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def main():
    # Load results
    with open("optuna_experiment_results.json", "r") as f:
        results = json.load(f)

    # Convert to RLiable format
    score_dict = {
        algo: np.array(scores)  # Shape: (num_seeds, num_episodes)
        for algo, scores in results.items()
    }

    print("Data loaded:")
    for algo, scores in score_dict.items():
        print(f"  {algo}: {scores.shape}")

    # Compute aggregate metrics
    aggregate_func = lambda x: np.array([
        metrics.aggregate_iqm(x),
        metrics.aggregate_median(x),
        metrics.aggregate_mean(x)
    ])

    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        score_dict,
        aggregate_func,
        reps=50000
    )

    algorithms = list(score_dict.keys())
    metric_names = ['IQM', 'Median', 'Mean']

    print("\n" + "="*70)
    print("  AGGREGATE METRICS WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    for i, metric in enumerate(metric_names):
        print(f"\n{metric}:")
        for algo in algorithms:
            score = aggregate_scores[algo][i]
            ci_low = aggregate_cis[algo][0, i]
            ci_high = aggregate_cis[algo][1, i]
            print(f"  {algo:<12} {score:7.2f}  [{ci_low:7.2f}, {ci_high:7.2f}]")

    # Performance profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_list = np.linspace(0, 500, 100)

    for algo_name, scores in score_dict.items():
        perf_prof = [(scores >= tau).mean() for tau in tau_list]
        ax.plot(tau_list, perf_prof, label=algo_name, linewidth=2.5)

    ax.set_xlabel("Score Threshold (τ)", fontsize=13)
    ax.set_ylabel("Fraction of Runs ≥ τ", fontsize=13)
    ax.set_title("Performance Profiles", fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("performance_profiles.png", dpi=300)
    print("\n✓ Saved: performance_profiles.png")

    plt.show()