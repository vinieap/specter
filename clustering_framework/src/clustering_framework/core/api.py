"""Public API for the clustering framework."""

from typing import List, Dict, Optional, Type, Any, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from ..algorithms import ClusteringAlgorithm
from .registry import algorithm_registry
from .types import OptimizationResult, ArrayLike
from ..optimizers import BatchOptimizer, SequentialOptimizer


def validate_array(
    X: ArrayLike, min_samples: int = 2, min_features: int = 1
) -> np.ndarray:
    """Validate input array for clustering.

    Parameters
    ----------
    X : array-like
        Input array to validate
    min_samples : int, default=2
        Minimum number of samples required
    min_features : int, default=1
        Minimum number of features required

    Returns
    -------
    np.ndarray
        Validated array

    Raises
    ------
    ValueError
        If array does not meet requirements
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    n_samples, n_features = X.shape
    if n_samples < min_samples:
        raise ValueError(f"At least {min_samples} samples required")
    if n_features < min_features:
        raise ValueError(f"At least {min_features} features required")

    return X


def get_algorithm(name: str, random_state: Optional[int] = None) -> ClusteringAlgorithm:
    """Get a clustering algorithm by name.

    Parameters
    ----------
    name : str
        Name of the algorithm to get
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    ClusteringAlgorithm
        Instance of the requested algorithm

    Raises
    ------
    ValueError
        If algorithm name is not registered
    """
    return algorithm_registry.get_algorithm(name, random_state)


def list_algorithms() -> List[str]:
    """Get a list of all available algorithm names.

    Returns
    -------
    List[str]
        List of algorithm names
    """
    return algorithm_registry.list_algorithms()


def get_algorithm_categories() -> Dict[str, List[str]]:
    """Get algorithms grouped by category.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping category names to lists of algorithm names
    """
    return algorithm_registry.get_algorithm_categories()


def register_algorithm(algorithm_class: Type[ClusteringAlgorithm]) -> None:
    """Register a new clustering algorithm.

    Parameters
    ----------
    algorithm_class : Type[ClusteringAlgorithm]
        The clustering algorithm class to register

    Raises
    ------
    ValueError
        If algorithm with same name is already registered
    TypeError
        If algorithm_class doesn't inherit from ClusteringAlgorithm
    """
    algorithm_registry.register_algorithm(algorithm_class)


def quick_cluster(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    algorithm: str = "kmeans",
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[BaseEstimator, Dict[str, float]]:
    """Quickly cluster data with automatic parameter selection.

    This function provides a simplified interface for clustering that:
    1. Automatically determines the optimal number of clusters if not specified
    2. Optimizes algorithm parameters
    3. Returns both the model and quality metrics

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset to cluster
    n_clusters : int, optional
        Number of clusters. If None, automatically determined
    max_clusters : int, default=10
        Maximum number of clusters to consider when n_clusters is None
    algorithm : str, default='kmeans'
        Clustering algorithm to use
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments passed to the algorithm

    Returns
    -------
    Tuple[BaseEstimator, Dict[str, float]]
        - Fitted clustering model
        - Dictionary of quality metrics

    Examples
    --------
    >>> from clustering_framework import quick_cluster
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> # Generate sample data
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>>
    >>> # Perform clustering
    >>> model, metrics = quick_cluster(X, n_clusters=4)
    >>> print(f"Silhouette score: {metrics['silhouette']:.3f}")
    """
    # Validate input array
    X = validate_array(X, min_samples=2, min_features=1)

    # If n_clusters not specified, find optimal number
    if n_clusters is None:
        # Run optimization with varying n_clusters
        best_score = float("-inf")
        best_n = 2

        for n in range(2, max_clusters + 1):
            # Run quick optimization
            results = optimize_clustering(
                X,
                algorithm=algorithm,
                n_calls=20,  # Reduced for speed
                random_state=random_state,
                n_clusters=n,
                **kwargs,
            )
            if results.best_score > best_score:
                best_score = results.best_score
                best_n = n

        n_clusters = best_n

    # Run optimization with chosen n_clusters (if algorithm supports it)
    algo = get_algorithm(algorithm)
    if hasattr(algo.estimator_class(), 'n_clusters'):
        results = optimize_clustering(
            X,
            algorithm=algorithm,
            n_calls=50,  # Reduced for speed
            random_state=random_state,
            n_clusters=n_clusters,
            **kwargs,
        )
    else:
        results = optimize_clustering(
            X,
            algorithm=algorithm,
            n_calls=50,  # Reduced for speed
            random_state=random_state,
            **kwargs,
        )

    # Evaluate results
    metrics = evaluate_clustering(
        X, results.best_model, metrics=["silhouette", "calinski_harabasz"]
    )

    return results.best_model, metrics


def evaluate_clustering(
    X: np.ndarray, model: BaseEstimator, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate clustering results using various metrics.

    This function computes quality metrics for a clustering result. It supports
    multiple metrics and handles edge cases like single-cluster results.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    model : BaseEstimator
        The fitted clustering model to evaluate
    metrics : List[str], optional
        List of metric names to compute. If None, uses default metrics.
        Available metrics:
        - 'silhouette': Silhouette coefficient
        - 'calinski_harabasz': Calinski-Harabasz index
        - 'davies_bouldin': Davies-Bouldin index

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to scores

    Examples
    --------
    >>> from clustering_framework import evaluate_clustering
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>>
    >>> # Generate sample data
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>>
    >>> # Fit clustering model
    >>> model = KMeans(n_clusters=4, random_state=42).fit(X)
    >>>
    >>> # Evaluate clustering
    >>> metrics = evaluate_clustering(
    ...     X,
    ...     model,
    ...     metrics=["silhouette", "calinski_harabasz"]
    ... )
    >>>
    >>> print("Silhouette score:", metrics["silhouette"])
    """
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )

    # Validate input array
    X = validate_array(X, min_samples=2, min_features=1)

    # Get labels using predict() or labels_ attribute
    if hasattr(model, "predict"):
        labels = model.predict(X)
    elif hasattr(model, "labels_"):
        labels = model.labels_
    else:
        raise ValueError(
            f"Model {type(model).__name__} has neither predict() method nor labels_ attribute"
        )

    # Check if we have enough clusters for scoring
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {metric: float("-inf") for metric in metrics} if metrics else {}

    # Use default metrics if none specified
    if metrics is None:
        metrics = ["silhouette"]

    # Map metric names to functions
    metric_funcs = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
    }

    # Validate requested metrics
    invalid_metrics = set(metrics) - set(metric_funcs.keys())
    if invalid_metrics:
        raise ValueError(
            f"Unknown metrics: {invalid_metrics}. "
            f"Available metrics: {list(metric_funcs.keys())}"
        )

    # Compute requested metrics
    results = {}
    for metric in metrics:
        try:
            score = metric_funcs[metric](X, labels)
            results[metric] = score
        except Exception as e:
            print(f"Warning: Failed to compute {metric} score: {str(e)}")
            results[metric] = float("-inf")

    return results


def analyze_clusters(
    X: np.ndarray,
    model: BaseEstimator,
    noise_analysis: bool = True,
    convergence_analysis: bool = True,
    stability_analysis: bool = False,
    n_stability_runs: int = 10,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze clustering results using various metrics and techniques.

    This function provides comprehensive analysis of clustering results,
    including noise detection, convergence analysis, and stability assessment.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    model : BaseEstimator
        The fitted clustering model to analyze
    noise_analysis : bool, default=True
        Whether to perform noise point analysis
    convergence_analysis : bool, default=True
        Whether to analyze convergence behavior
    stability_analysis : bool, default=False
        Whether to assess clustering stability through multiple runs
    n_stability_runs : int, default=10
        Number of runs for stability analysis
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Analysis results including:
        - noise_analysis: NoiseAnalysis if enabled
        - convergence_info: Dict[str, Any] if enabled
        - stability_scores: Dict[str, float] if enabled

    Examples
    --------
    >>> from clustering_framework import optimize_clustering, analyze_clusters
    >>>
    >>> # First optimize clustering
    >>> results = optimize_clustering(X, algorithm="dbscan")
    >>>
    >>> # Then analyze the results
    >>> analysis = analyze_clusters(
    ...     X,
    ...     results.best_model,
    ...     stability_analysis=True
    ... )
    >>>
    >>> # Print noise analysis
    >>> if analysis["noise_analysis"]:
    ...     print(f"Noise ratio: {analysis['noise_analysis'].noise_ratio:.2f}")
    ...     print("Recommendations:", analysis["noise_analysis"].recommendations)
    """
    from ..analysis.noise import analyze_noise
    from ..analysis.stability import analyze_stability
    from ..analysis.convergence import analyze_convergence

    # Validate input array
    X = validate_array(X, min_samples=2, min_features=1)

    results = {}

    if noise_analysis:
        results["noise_analysis"] = analyze_noise(X, model, random_state=random_state)

    if convergence_analysis:
        results["convergence_info"] = analyze_convergence(model, X)

    if stability_analysis:
        results["stability_scores"] = analyze_stability(
            X, model, n_splits=n_stability_runs, random_state=random_state
        )

    return results


def optimize_clustering(
    X: ArrayLike,
    algorithm: str = "kmeans",
    n_calls: int = 100,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None,
    use_batch_optimizer: bool = True,
    use_dashboard: bool = False,
    dashboard_port: int = 8080,
    random_state: Optional[int] = None,
    **optimizer_kwargs: Any,
) -> OptimizationResult:
    """Optimize clustering parameters for a dataset using Bayesian optimization.

    This function provides a high-level interface for optimizing clustering
    parameters using either parallel batch optimization or sequential
    optimization. It supports all clustering algorithms registered in the
    AlgorithmRegistry.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset to cluster
    algorithm : str, default='kmeans'
        The clustering algorithm to optimize
    n_calls : int, default=100
        Number of optimization iterations
    n_jobs : int, optional
        Number of parallel jobs (defaults to cpu_count() - 1)
        Only used if use_batch_optimizer=True
    batch_size : int, optional
        Batch size for parallel evaluation (defaults to n_jobs * 2, max 8)
        Only used if use_batch_optimizer=True
    use_batch_optimizer : bool, default=True
        Whether to use parallel batch optimization (recommended for large datasets)
        or sequential optimization (better for small datasets or debugging)
    use_dashboard : bool, default=False
        Whether to start optuna-dashboard for real-time visualization
    dashboard_port : int, default=8080
        Port for the optuna-dashboard web interface
    random_state : int, optional
        Random seed for reproducibility
    **optimizer_kwargs : dict
        Additional arguments passed to the optimizer

    Returns
    -------
    OptimizationResult
        A dataclass containing:
        - best_score: float, the best achieved clustering score
        - best_params: Dict[str, Any], the optimal parameters
        - best_model: BaseEstimator, the fitted clustering model
        - history: List[Dict[str, Any]], optimization history
        - convergence_info: Dict[str, Any], convergence statistics
        - execution_stats: Dict[str, Any], runtime statistics

    Examples
    --------
    >>> from clustering_framework import optimize_clustering
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> # Generate sample data
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>>
    >>> # Optimize KMeans clustering
    >>> results = optimize_clustering(
    ...     X,
    ...     algorithm="kmeans",
    ...     n_calls=50,
    ...     use_dashboard=True
    ... )
    >>>
    >>> print(f"Best score: {results.best_score}")
    >>> print(f"Best parameters: {results.best_params}")
    """
    # Validate input array
    X = validate_array(X, min_samples=2, min_features=1)

    # Get algorithm instance
    algo = get_algorithm(algorithm, random_state=random_state)

    # Choose optimizer class
    optimizer_class = BatchOptimizer if use_batch_optimizer else SequentialOptimizer

    # Set algorithm parameters
    if optimizer_kwargs:
        algo.set_parameters(optimizer_kwargs)

    # Create optimizer
    optimizer = optimizer_class(
        algorithm=algo,
        max_trials=n_calls,
        n_jobs=n_jobs,
        batch_size=batch_size,
        use_dashboard=use_dashboard,
        dashboard_port=dashboard_port,
        random_state=random_state,
    )

    # Run optimization
    return optimizer.optimize(X)
