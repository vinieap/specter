"""Example usage of different clustering algorithms."""
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

from .api import optimize_clustering
from .algorithms.registry import algorithm_registry


def example_kmeans():
    """Example of optimizing K-Means clustering."""
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    X = StandardScaler().fit_transform(X)

    # Optimize K-Means clustering
    results = optimize_clustering(
        X,
        algorithm="kmeans",
        n_calls=50,
        use_dashboard=True,
    )

    print("\nK-Means Clustering Results:")
    print(f"Best score: {results['best_score']:.4f}")
    print("Best parameters:", results["best_params"])


def example_dbscan():
    """Example of optimizing DBSCAN clustering."""
    # Generate sample data
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X)

    # Optimize DBSCAN clustering
    results = optimize_clustering(
        X,
        algorithm="dbscan",
        n_calls=50,
        use_dashboard=True,
    )

    print("\nDBSCAN Clustering Results:")
    print(f"Best score: {results['best_score']:.4f}")
    print("Best parameters:", results["best_params"])


def example_hdbscan():
    """Example of optimizing HDBSCAN clustering."""
    # Generate sample data with varying densities
    X1, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=0)
    X2, _ = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=1)
    X = np.vstack([X1, X2])
    X = StandardScaler().fit_transform(X)

    # Optimize HDBSCAN clustering
    results = optimize_clustering(
        X,
        algorithm="hdbscan",
        n_calls=50,
        use_dashboard=True,
    )

    print("\nHDBSCAN Clustering Results:")
    print(f"Best score: {results['best_score']:.4f}")
    print("Best parameters:", results["best_params"])


def example_affinity_propagation():
    """Example of optimizing Affinity Propagation clustering."""
    # Generate sample data
    X, _ = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X)

    # Optimize Affinity Propagation clustering
    results = optimize_clustering(
        X,
        algorithm="affinity_propagation",
        n_calls=50,
        use_dashboard=True,
    )

    print("\nAffinity Propagation Clustering Results:")
    print(f"Best score: {results['best_score']:.4f}")
    print("Best parameters:", results["best_params"])


def example_spectral():
    """Example of optimizing Spectral clustering."""
    # Generate sample data
    X, _ = make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=0)
    X = StandardScaler().fit_transform(X)

    # Optimize Spectral clustering
    results = optimize_clustering(
        X,
        algorithm="spectral",
        n_calls=50,
        use_dashboard=True,
    )

    print("\nSpectral Clustering Results:")
    print(f"Best score: {results['best_score']:.4f}")
    print("Best parameters:", results["best_params"])


def list_available_algorithms():
    """List all available clustering algorithms."""
    print("\nAvailable Clustering Algorithms:")
    for algo in algorithm_registry.list_algorithms():
        print(f"- {algo}")


def example_multi_study():
    """Example of running multiple studies with different algorithms and seeds."""
    # Generate sample data with multiple cluster shapes
    n_samples = 300
    
    # Create clusters with different shapes
    X1, _ = make_blobs(n_samples=n_samples//3, centers=2, cluster_std=0.5, random_state=0)
    X2, _ = make_moons(n_samples=n_samples//3, noise=0.05, random_state=0)
    X3, _ = make_circles(n_samples=n_samples//3, noise=0.05, factor=0.5, random_state=0)
    
    # Combine datasets
    X = np.vstack([X1, X2, X3])
    X = StandardScaler().fit_transform(X)
    
    # Run multi-study optimization
    from .api import optimize_clustering_multi_study
    
    # Select a subset of algorithms to compare
    algorithms = ["kmeans", "dbscan", "spectral", "hdbscan"]
    
    results = optimize_clustering_multi_study(
        X,
        n_seeds=3,  # Use 3 different random seeds
        algorithms=algorithms,  # Compare these algorithms
        n_calls=30,  # Fewer calls for example
        verbosity=2,  # Show detailed progress
    )
    
    # Results are automatically printed during optimization
    # You can also access specific metrics:
    for algorithm, metrics in results.items():
        print(f"\nDetailed metrics for {algorithm}:")
        print(f"Score stability: {metrics.score_stability:.4f}")
        print("Top 3 important parameters:")
        for param, importance in sorted(
            metrics.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            print(f"  {param}: {importance:.3f}")


if __name__ == "__main__":
    # List available algorithms
    list_available_algorithms()

    print("\nRunning clustering optimization examples...")
    
    # Run individual algorithm examples
    example_kmeans()
    example_dbscan()
    example_hdbscan()
    example_affinity_propagation()
    example_spectral()
    
    # Run multi-study example
    print("\nRunning multi-study optimization example...")
    example_multi_study()
    
    print("\nAll examples completed!")