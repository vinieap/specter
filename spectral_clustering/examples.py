import matplotlib.pyplot as plt
import numpy as np
from optuna import visualization as optuna_vis
from sklearn.datasets import make_blobs

from .api import optimize_spectral_clustering
from .config import N_CORES, VerbosityLevel
from .visualization import (
    create_optimization_summary_plots,
    generate_optimization_visualizations,
)


def run_performance_comparison():
    """Example usage with performance comparison between batch and sequential optimizers"""
    # Generate sample data
    X, y = make_blobs(n_samples=400, n_features=6, centers=4, random_state=42)

    # Same number of evaluations for fair comparison
    n_evaluations = 60

    print(f"System: {N_CORES} parallel cores available")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Evaluations: {n_evaluations} total for each optimizer\n")

    print("=== Batch Bayesian Optimization (Ask-and-Tell API) ===")
    # Optimize batch size for the system
    optimal_batch_size = min(N_CORES, 12)  # Don't exceed 12 for memory efficiency

    batch_results = optimize_spectral_clustering(
        X,
        n_calls=n_evaluations,
        batch_size=optimal_batch_size,
        verbosity=VerbosityLevel.MEDIUM,
        use_batch_optimizer=True,
        use_dashboard=True,  # Enable dashboard for demonstration
        dashboard_port=8080,
    )

    print("\nBatch Results:")
    print(f"🏆 Best silhouette score: {batch_results['best_score']:.4f}")
    print(f"⏱️  Optimization time: {batch_results['optimization_time']:.2f}s")
    print(f"🚀 Evaluations/sec: {batch_results['evaluations_per_second']:.1f}")
    print(f"📊 Total evaluations: {batch_results['n_evaluations']}")
    print(f"📦 Batch size used: {optimal_batch_size}")

    # Use the optimized clusterer
    clusterer = batch_results["best_clusterer"]
    labels = clusterer.fit_predict(X)
    print(f"🎯 Final clustering: {len(np.unique(labels))} clusters")

    print("\n=== Sequential Optimization (Traditional gp_minimize) ===")
    sequential_results = optimize_spectral_clustering(
        X,
        n_calls=n_evaluations,  # Same number of evaluations
        verbosity=VerbosityLevel.MEDIUM,
        use_batch_optimizer=False,
        use_dashboard=False,  # Don't start another dashboard instance
        dashboard_port=8080,
    )

    print("\nSequential Results:")
    print(f"🏆 Best silhouette score: {sequential_results['best_score']:.4f}")
    print(f"⏱️  Optimization time: {sequential_results['optimization_time']:.2f}s")
    print(
        f"🚀 Evaluations/sec: {sequential_results['n_evaluations'] / sequential_results['optimization_time']:.1f}"
    )
    print(f"📊 Total evaluations: {sequential_results['n_evaluations']}")

    # Performance comparison
    time_ratio = (
        batch_results["optimization_time"] / sequential_results["optimization_time"]
    )
    speedup = 1 / time_ratio if time_ratio < 1 else -(time_ratio - 1)

    print("\n=== Performance Summary ===")
    print(
        f"⏱️  Time comparison: Batch {batch_results['optimization_time']:.1f}s vs Sequential {sequential_results['optimization_time']:.1f}s"
    )
    print(
        f"🚀 Speed difference: {abs(speedup):.1f}x {'faster' if speedup > 0 else 'slower'}"
    )
    print(
        f"🎯 Score comparison: Batch {batch_results['best_score']:.4f} vs Sequential {sequential_results['best_score']:.4f}"
    )
    print(
        f"📈 Score difference: {(batch_results['best_score'] - sequential_results['best_score']):.4f}"
    )

    # Analysis
    print("\n💡 Analysis:")
    if batch_results["best_score"] > sequential_results["best_score"] + 0.01:
        print("  ✓ Batch optimizer found better parameters (higher score)")
    elif sequential_results["best_score"] > batch_results["best_score"] + 0.01:
        print("  ✓ Sequential optimizer found better parameters (higher score)")
    else:
        print("  ✓ Both optimizers found comparable solutions")

    if speedup > 1.2:
        print("  ✓ Batch optimizer is significantly faster")
    elif speedup < -1.2:
        print(
            "  ✓ Sequential optimizer is faster (batch overhead not worth it for this problem)"
        )
    else:
        print("  ✓ Similar execution times")

    print("\n📝 Recommendation: ", end="")
    if (
        batch_results["best_score"] >= sequential_results["best_score"]
        and speedup > -0.5
    ):
        print("Use batch optimizer for better exploration and parallelization")
    elif X.shape[0] < 200 or n_evaluations < 50:
        print("Sequential optimizer may be sufficient for small datasets/evaluations")
    else:
        print(
            "Consider problem-specific factors (cluster evaluation cost, available cores)"
        )

    print("\n💡 Dashboard Usage:")
    print("   The example above demonstrates optuna-dashboard integration.")
    print("   When use_dashboard=True, you can monitor optimization in real-time at:")
    print("   - Dashboard URL: http://localhost:8080")
    print("   - Database: optuna_spectral_studies.db (contains all studies)")
    print("   - Both batch and sequential optimizers are stored in the same database")
    print("   Install with: pip install optuna-dashboard")

    # Generate visualizations
    _create_example_visualizations(batch_results, sequential_results)

    return batch_results, sequential_results


def _create_example_visualizations(batch_results, sequential_results):
    """Create comprehensive visualizations for the example"""
    print("\n\n=== Generating Optimization Visualizations ===")

    # For sequential optimizer (has Optuna study)
    if "study" in sequential_results:
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
            print("  ✓ Saved: sequential_optimization_history.png")
        except Exception as e:
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
            print("  ✓ Saved: sequential_param_importance.png")
        except Exception as e:
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
            print("  ✓ Saved: sequential_parallel_coordinate.png")
        except Exception as e:
            print(f"  ⚠ Could not create parallel coordinate plot: {e}")

    # For batch optimizer
    if "study" in batch_results:
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
            print("  ✓ Saved: batch_optimization_history.png")
        except Exception as e:
            print(f"  ⚠ Could not create optimization history plot: {e}")

        # Convergence comparison plot using matplotlib
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 1, 1)

        # Extract convergence data from both studies
        if "study" in sequential_results:
            seq_trials = sequential_results["study"].trials
            seq_values = [trial.value for trial in seq_trials]
            seq_best = np.maximum.accumulate(seq_values)  # Maximize silhouette score
            ax.plot(seq_best, "b-", linewidth=2, label="Sequential")

        batch_trials = batch_results["study"].trials
        batch_values = [trial.value for trial in batch_trials]
        batch_best = np.maximum.accumulate(batch_values)  # Maximize silhouette score
        ax.plot(batch_best, "g-", linewidth=2, label="Batch")

        ax.set_title("Optimization Convergence Comparison", fontsize=16)
        ax.set_ylabel("Best Silhouette Score", fontsize=12)
        ax.set_xlabel("Number of Evaluations", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("convergence_comparison.png", dpi=150, bbox_inches="tight")
        print("  ✓ Saved: convergence_comparison.png")

    # Create summary plots
    create_optimization_summary_plots(batch_results, sequential_results)

    print("\n📊 Visualization files created:")
    print("  • convergence_comparison.png - Shows optimization convergence")
    print("  • optimization_summary.png - Detailed analysis of both optimizers")
    if "study" in sequential_results:
        print("  • sequential_optimization_history.png - Optimization history plot")
        print("  • sequential_param_importance.png - Parameter importance analysis")
        print(
            "  • sequential_parallel_coordinate.png - Parallel coordinate visualization"
        )
    if "study" in batch_results:
        print("  • batch_optimization_history.png - Batch optimization history")

    plt.show()


if __name__ == "__main__":
    run_performance_comparison()
