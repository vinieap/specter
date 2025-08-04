"""
Clustering Library Public API

This module provides the main public interface for the clustering library.
It includes high-level functions for clustering optimization, analysis,
and evaluation, as well as convenience functions for common workflows.

Example:
    >>> from clustering import optimize_clustering, analyze_clusters
    >>> from sklearn.datasets import make_blobs
    >>> 
    >>> # Generate sample data
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>> 
    >>> # Optimize clustering parameters
    >>> results = optimize_clustering(X, algorithm="kmeans", n_calls=50)
    >>> 
    >>> # Analyze the clustering results
    >>> analysis = analyze_clusters(X, results.best_model)
"""

import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from .core.types import OptimizationResult, NoiseAnalysis, AlgorithmPerformance
from .optimizers import (
    BatchBayesianClusteringOptimizer,
    SequentialClusteringOptimizer,
    MultiStudyOptimizer
)
from .analysis import NoiseAnalyzer, ConvergenceAnalyzer, ClusterEvaluator
from .core.registry import AlgorithmRegistry
from .config import DEFAULT_VERBOSITY, VerbosityLevel
from .validation import (
    validate_array,
    validate_estimator,
    validate_algorithm,
    validate_optimizer_params,
    validate_analysis_params,
    validate_metrics
)


def optimize_clustering(
    X: np.ndarray,
    algorithm: str = "kmeans",
    n_calls: int = 100,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None,
    verbosity: int = DEFAULT_VERBOSITY,
    use_batch_optimizer: bool = True,
    use_dashboard: bool = False,
    dashboard_port: int = 8080,
    random_state: Optional[int] = None,
    **optimizer_kwargs
) -> OptimizationResult:
    """
    Optimize clustering parameters for a dataset using Bayesian optimization.

    This function provides a high-level interface for optimizing clustering
    parameters using either parallel batch optimization or sequential
    optimization. It supports all clustering algorithms registered in the
    AlgorithmRegistry.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset to cluster
    algorithm : str, default='kmeans'
        The clustering algorithm to optimize. One of:
        'kmeans', 'mini_batch_kmeans', 'bisecting_kmeans',
        'spectral', 'dbscan', 'hdbscan', 'optics',
        'agglomerative', 'birch', 'affinity_propagation'
    n_calls : int, default=100
        Number of optimization iterations
    n_jobs : int, optional
        Number of parallel jobs (defaults to cpu_count() - 1)
        Only used if use_batch_optimizer=True
    batch_size : int, optional
        Batch size for parallel evaluation (defaults to n_jobs * 2, max 8)
        Only used if use_batch_optimizer=True
    verbosity : int, default=VerbosityLevel.DETAILED
        Verbosity level for output
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
    >>> from clustering import optimize_clustering
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

    See Also
    --------
    analyze_clusters : Analyze clustering results
    evaluate_clustering : Evaluate clustering quality
    compare_algorithms : Compare multiple clustering algorithms
    """
    # Validate input array
    X = validate_array(X, min_samples=2, min_features=1)

    # Set random state
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Validate algorithm choice
    algorithm = validate_algorithm(
        algorithm,
        available_algorithms=AlgorithmRegistry.list_algorithms()
    )

    # Validate optimizer parameters
    optimizer_params = validate_optimizer_params(
        n_calls=n_calls,
        n_jobs=n_jobs,
        batch_size=batch_size,
        verbosity=verbosity,
        use_batch_optimizer=use_batch_optimizer,
        use_dashboard=use_dashboard,
        dashboard_port=dashboard_port
    )

    # Create optimizer
    if use_batch_optimizer:
        optimizer = BatchBayesianClusteringOptimizer(
            algorithm=algorithm,
            n_calls=n_calls,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbosity=verbosity,
            use_dashboard=use_dashboard,
            dashboard_port=dashboard_port,
            random_state=random_state,
            **optimizer_kwargs
        )
    else:
        optimizer = SequentialClusteringOptimizer(
            algorithm=algorithm,
            n_calls=n_calls,
            verbosity=verbosity,
            use_dashboard=use_dashboard,
            dashboard_port=dashboard_port,
            random_state=random_state,
            **optimizer_kwargs
        )

    # Run optimization
    return optimizer.optimize(X)


def analyze_clusters(
    X: np.ndarray,
    model: BaseEstimator,
    noise_analysis: bool = True,
    convergence_analysis: bool = True,
    stability_analysis: bool = False,
    n_stability_runs: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze clustering results using various metrics and techniques.

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
    >>> from clustering import optimize_clustering, analyze_clusters
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
    # Validate input array if noise analysis is enabled
    if noise_analysis:
        X = validate_array(X, min_samples=1, min_features=1)

    # Validate model
    model = validate_estimator(model)

    # Validate analysis parameters
    analysis_params = validate_analysis_params(
        noise_analysis=noise_analysis,
        convergence_analysis=convergence_analysis,
        stability_analysis=stability_analysis,
        n_stability_runs=n_stability_runs
    )

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    results = {}

    if noise_analysis:
        analyzer = NoiseAnalyzer(random_state=random_state)
        results["noise_analysis"] = analyzer.analyze(X, model)

    if convergence_analysis:
        analyzer = ConvergenceAnalyzer()
        results["convergence_info"] = analyzer.analyze(model)

    if stability_analysis:
        results["stability_scores"] = _analyze_stability(
            X, model, n_runs=n_stability_runs, random_state=random_state
        )

    return results


def evaluate_clustering(
    X: np.ndarray,
    model: BaseEstimator,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate clustering quality using various metrics.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    model : BaseEstimator
        The fitted clustering model to evaluate
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.
        Available metrics: 'silhouette', 'calinski_harabasz', 'davies_bouldin'

    Returns
    -------
    Dict[str, float]
        Dictionary mapping metric names to their values

    Examples
    --------
    >>> from clustering import optimize_clustering, evaluate_clustering
    >>> 
    >>> # Optimize clustering
    >>> results = optimize_clustering(X, algorithm="kmeans")
    >>> 
    >>> # Evaluate the results
    >>> scores = evaluate_clustering(
    ...     X,
    ...     results.best_model,
    ...     metrics=['silhouette', 'calinski_harabasz']
    ... )
    >>> 
    >>> for metric, score in scores.items():
    ...     print(f"{metric}: {score:.3f}")
    """
    # Validate input array
    X = validate_array(X, min_samples=2, min_features=1)

    # Validate model
    model = validate_estimator(model)

    # Validate metrics
    available_metrics = [
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin'
    ]
    metrics = validate_metrics(metrics, available_metrics)

    evaluator = ClusterEvaluator()
    return evaluator.evaluate(X, model, metrics=metrics)


def compare_algorithms(
    X: np.ndarray,
    algorithms: List[str],
    n_calls: int = 50,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **optimizer_kwargs
) -> Dict[str, AlgorithmPerformance]:
    """
    Compare multiple clustering algorithms on a dataset.

    This function runs optimization for multiple algorithms and compares
    their performance using various metrics.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    algorithms : List[str]
        List of algorithm names to compare
    n_calls : int, default=50
        Number of optimization calls per algorithm
    n_jobs : int, optional
        Number of parallel jobs
    random_state : int, optional
        Random seed for reproducibility
    **optimizer_kwargs
        Additional arguments passed to optimize_clustering

    Returns
    -------
    Dict[str, AlgorithmPerformance]
        Performance metrics and results for each algorithm

    Examples
    --------
    >>> from clustering import compare_algorithms
    >>> 
    >>> # Compare multiple algorithms
    >>> results = compare_algorithms(
    ...     X,
    ...     algorithms=['kmeans', 'dbscan', 'spectral'],
    ...     n_calls=50
    ... )
    >>> 
    >>> # Print comparison results
    >>> for algo, perf in results.items():
    ...     print(f"{algo}:")
    ...     print(f"  Best score: {perf.best_score:.3f}")
    ...     print(f"  Training time: {perf.training_time:.2f}s")
    ...     print(f"  Memory usage: {perf.memory_usage_mb:.1f}MB")
    """
    optimizer = MultiStudyOptimizer(
        algorithms=algorithms,
        n_calls=n_calls,
        n_jobs=n_jobs,
        random_state=random_state,
        **optimizer_kwargs
    )
    return optimizer.compare(X)


def _analyze_stability(
    X: np.ndarray,
    model: BaseEstimator,
    n_runs: int = 10,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """Helper function for stability analysis."""
    if random_state is not None:
        np.random.seed(random_state)

    # Implementation details...
    pass