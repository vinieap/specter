import matplotlib.pyplot as plt
import numpy as np
from optuna import visualization as optuna_vis

from .config import VerbosityLevel


def generate_optimization_visualizations(
    batch_results=None, sequential_results=None, verbosity=VerbosityLevel.MINIMAL
):
    """
    Generate optimization visualizations from study results.

    Parameters:
    -----------
    batch_results : dict, optional
        Results from BatchBayesianSpectralOptimizer.optimize()
    sequential_results : dict, optional
        Results from BayesianSpectralOptimizer.optimize()
    verbosity : int, default=VerbosityLevel.MINIMAL
        Verbosity level for output

    Returns:
    --------
    list : List of generated visualization file names
    """
    generated_files = []

    if verbosity >= VerbosityLevel.MINIMAL:
        print("\n=== Generating Optimization Visualizations ===")

    # For sequential optimizer (has Optuna study)
    if sequential_results and "study" in sequential_results:
        if verbosity >= VerbosityLevel.MINIMAL:
            print("Creating plots for sequential optimizer...")

        try:
            # 1. Optimization History Plot
            fig = optuna_vis.plot_optimization_history(sequential_results["study"])
            fig.update_layout(
                title="Sequential Optimizer: Optimization History",
                width=1000,
                height=600,
            )
            fig.write_image("sequential_optimization_history.png")
            generated_files.append("sequential_optimization_history.png")
            if verbosity >= VerbosityLevel.MINIMAL:
                print("  ✓ Saved: sequential_optimization_history.png")
        except Exception as e:
            if verbosity >= VerbosityLevel.MINIMAL:
                print(f"  ⚠ Could not create optimization history plot: {e}")

        try:
            # 2. Parameter Importance
            fig = optuna_vis.plot_param_importances(sequential_results["study"])
            fig.update_layout(
                title="Sequential Optimizer: Parameter Importance",
                width=1000,
                height=600,
            )
            fig.write_image("sequential_param_importance.png")
            generated_files.append("sequential_param_importance.png")
            if verbosity >= VerbosityLevel.MINIMAL:
                print("  ✓ Saved: sequential_param_importance.png")
        except Exception as e:
            if verbosity >= VerbosityLevel.MINIMAL:
                print(f"  ⚠ Could not create parameter importance plot: {e}")

        try:
            # 3. Parallel Coordinate Plot
            fig = optuna_vis.plot_parallel_coordinate(sequential_results["study"])
            fig.update_layout(
                title="Sequential Optimizer: Parallel Coordinates",
                width=1200,
                height=800,
            )
            fig.write_image("sequential_parallel_coordinate.png")
            generated_files.append("sequential_parallel_coordinate.png")
            if verbosity >= VerbosityLevel.MINIMAL:
                print("  ✓ Saved: sequential_parallel_coordinate.png")
        except Exception as e:
            if verbosity >= VerbosityLevel.MINIMAL:
                print(f"  ⚠ Could not create parallel coordinate plot: {e}")

    # For batch optimizer
    if batch_results and "study" in batch_results:
        if verbosity >= VerbosityLevel.MINIMAL:
            print("\nCreating plots for batch optimizer...")

        try:
            # 1. Optimization History Plot
            fig = optuna_vis.plot_optimization_history(batch_results["study"])
            fig.update_layout(
                title="Batch Optimizer: Optimization History",
                width=1000,
                height=600,
            )
            fig.write_image("batch_optimization_history.png")
            generated_files.append("batch_optimization_history.png")
            if verbosity >= VerbosityLevel.MINIMAL:
                print("  ✓ Saved: batch_optimization_history.png")
        except Exception as e:
            if verbosity >= VerbosityLevel.MINIMAL:
                print(f"  ⚠ Could not create optimization history plot: {e}")

        try:
            # 2. Parameter Importance
            fig = optuna_vis.plot_param_importances(batch_results["study"])
            fig.update_layout(
                title="Batch Optimizer: Parameter Importance",
                width=1000,
                height=600,
            )
            fig.write_image("batch_param_importance.png")
            generated_files.append("batch_param_importance.png")
            if verbosity >= VerbosityLevel.MINIMAL:
                print("  ✓ Saved: batch_param_importance.png")
        except Exception as e:
            if verbosity >= VerbosityLevel.MINIMAL:
                print(f"  ⚠ Could not create batch parameter importance plot: {e}")

        try:
            # 3. Parallel Coordinate Plot
            fig = optuna_vis.plot_parallel_coordinate(batch_results["study"])
            fig.update_layout(
                title="Batch Optimizer: Parallel Coordinates",
                width=1200,
                height=800,
            )
            fig.write_image("batch_parallel_coordinate.png")
            generated_files.append("batch_parallel_coordinate.png")
            if verbosity >= VerbosityLevel.MINIMAL:
                print("  ✓ Saved: batch_parallel_coordinate.png")
        except Exception as e:
            if verbosity >= VerbosityLevel.MINIMAL:
                print(f"  ⚠ Could not create batch parallel coordinate plot: {e}")

    # Convergence comparison plot using matplotlib
    if (
        batch_results
        and "study" in batch_results
        and sequential_results
        and "study" in sequential_results
    ):
        if verbosity >= VerbosityLevel.MINIMAL:
            print("\nCreating convergence comparison...")

        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 1, 1)

        # Extract convergence data from both studies
        seq_trials = sequential_results["study"].trials
        seq_values = [trial.value for trial in seq_trials if trial.value is not None]
        if seq_values:
            seq_best = np.maximum.accumulate(seq_values)  # Maximize silhouette score
            ax.plot(seq_best, "b-", linewidth=2, label="Sequential")

        batch_trials = batch_results["study"].trials
        batch_values = [
            trial.value for trial in batch_trials if trial.value is not None
        ]
        if batch_values:
            batch_best = np.maximum.accumulate(
                batch_values
            )  # Maximize silhouette score
            ax.plot(batch_best, "g-", linewidth=2, label="Batch")

        ax.set_title("Optimization Convergence Comparison", fontsize=16)
        ax.set_ylabel("Best Silhouette Score", fontsize=12)
        ax.set_xlabel("Number of Evaluations", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("convergence_comparison.png", dpi=150, bbox_inches="tight")
        generated_files.append("convergence_comparison.png")
        if verbosity >= VerbosityLevel.MINIMAL:
            print("  ✓ Saved: convergence_comparison.png")
        plt.close()

    if verbosity >= VerbosityLevel.MINIMAL:
        print(f"\n📊 Generated {len(generated_files)} visualization files!")

    return generated_files


def create_optimization_summary_plots(batch_results, sequential_results):
    """Create detailed summary plots comparing both optimizers"""
    # Summary statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Score progression
    ax = axes[0, 0]
    seq_scores = [
        -eval["score"]
        for eval in sequential_results["evaluation_history"]
        if eval["success"]
    ]
    batch_scores = [
        -eval["score"]
        for eval in batch_results["evaluation_history"]
        if eval["success"]
    ]

    ax.plot(seq_scores, "b-", alpha=0.5, label="Sequential")
    ax.plot(batch_scores, "g-", alpha=0.5, label="Batch")
    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel("Silhouette Score (negative)")
    ax.set_title("All Evaluation Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Best score progression
    ax = axes[0, 1]
    seq_best = np.minimum.accumulate(seq_scores)
    batch_best = np.minimum.accumulate(batch_scores)

    ax.plot(seq_best, "b-", linewidth=2, label="Sequential")
    ax.plot(batch_best, "g-", linewidth=2, label="Batch")
    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel("Best Score Found (negative)")
    ax.set_title("Best Score Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Parameter distribution (n_clusters)
    ax = axes[1, 0]
    seq_clusters = [
        eval["params"]["n_clusters"]
        for eval in sequential_results["evaluation_history"]
        if eval["success"]
    ]
    batch_clusters = [
        eval["params"]["n_clusters"]
        for eval in batch_results["evaluation_history"]
        if eval["success"]
    ]

    ax.hist(seq_clusters, bins=15, alpha=0.5, label="Sequential", color="blue")
    ax.hist(batch_clusters, bins=15, alpha=0.5, label="Batch", color="green")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Frequency")
    ax.set_title("Explored n_clusters Distribution")
    ax.legend()

    # 4. Affinity distribution
    ax = axes[1, 1]
    seq_affinity = [
        eval["params"]["affinity"]
        for eval in sequential_results["evaluation_history"]
        if eval["success"]
    ]
    batch_affinity = [
        eval["params"]["affinity"]
        for eval in batch_results["evaluation_history"]
        if eval["success"]
    ]

    affinity_types = ["rbf", "polynomial", "nearest_neighbors"]
    seq_counts = [seq_affinity.count(a) for a in affinity_types]
    batch_counts = [batch_affinity.count(a) for a in affinity_types]

    x = np.arange(len(affinity_types))
    width = 0.35

    ax.bar(
        x - width / 2, seq_counts, width, label="Sequential", color="blue", alpha=0.7
    )
    ax.bar(x + width / 2, batch_counts, width, label="Batch", color="green", alpha=0.7)
    ax.set_xlabel("Affinity Type")
    ax.set_ylabel("Times Evaluated")
    ax.set_title("Affinity Type Exploration")
    ax.set_xticks(x)
    ax.set_xticklabels(affinity_types)
    ax.legend()

    plt.suptitle("Optimization Analysis Summary", fontsize=16)
    plt.tight_layout()
    plt.savefig("optimization_summary.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved: optimization_summary.png")

    return "optimization_summary.png"
