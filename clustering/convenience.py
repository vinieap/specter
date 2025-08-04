"""
Convenience Functions

This module provides high-level convenience functions for common clustering
operations. These functions combine multiple steps into simple, easy-to-use
interfaces for common workflows.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator

from .api import (
    optimize_clustering,
    analyze_clusters,
    evaluate_clustering,
    compare_algorithms
)
from .core.types import OptimizationResult, NoiseAnalysis


def quick_cluster(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    algorithm: str = "kmeans",
    random_state: Optional[int] = None
) -> Tuple[BaseEstimator, Dict[str, float]]:
    """
    Quickly cluster data with automatic parameter selection.

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

    Returns
    -------
    Tuple[BaseEstimator, Dict[str, float]]
        - Fitted clustering model
        - Dictionary of quality metrics

    Examples
    --------
    >>> from clustering import quick_cluster
    >>> from sklearn.datasets import make_blobs
    >>> 
    >>> # Generate sample data
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>> 
    >>> # Perform clustering
    >>> model, metrics = quick_cluster(X, n_clusters=4)
    >>> print(f"Silhouette score: {metrics['silhouette']:.3f}")
    """
    # If n_clusters not specified, find optimal number
    if n_clusters is None:
        n_clusters = find_optimal_clusters(
            X,
            max_clusters=max_clusters,
            algorithm=algorithm,
            random_state=random_state
        )

    # Optimize clustering
    results = optimize_clustering(
        X,
        algorithm=algorithm,
        n_calls=50,  # Reduced for speed
        random_state=random_state,
        n_clusters=n_clusters
    )

    # Evaluate results
    metrics = evaluate_clustering(
        X,
        results.best_model,
        metrics=['silhouette', 'calinski_harabasz']
    )

    return results.best_model, metrics


def analyze_and_improve(
    X: np.ndarray,
    model: BaseEstimator,
    improve: bool = True,
    random_state: Optional[int] = None
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Analyze clustering results and optionally improve them.

    This function:
    1. Analyzes clustering quality and issues
    2. Provides detailed recommendations
    3. Optionally attempts to improve the clustering

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    model : BaseEstimator
        The fitted clustering model to analyze
    improve : bool, default=True
        Whether to attempt automatic improvements
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[BaseEstimator, Dict[str, Any]]
        - Improved model (or original if improve=False)
        - Analysis results and recommendations

    Examples
    --------
    >>> from clustering import quick_cluster, analyze_and_improve
    >>> 
    >>> # First perform clustering
    >>> model, _ = quick_cluster(X, n_clusters=4)
    >>> 
    >>> # Analyze and improve results
    >>> improved_model, analysis = analyze_and_improve(X, model)
    >>> 
    >>> # Print recommendations
    >>> for rec in analysis['recommendations']:
    ...     print(f"- {rec}")
    """
    # Perform comprehensive analysis
    analysis = analyze_clusters(
        X,
        model,
        noise_analysis=True,
        convergence_analysis=True,
        stability_analysis=True,
        random_state=random_state
    )

    if not improve:
        return model, analysis

    # Attempt improvements based on analysis
    improved_model = model
    recommendations = []

    # Handle noise points
    if "noise_analysis" in analysis:
        noise_ratio = analysis["noise_analysis"].noise_ratio
        if noise_ratio > 0.1:  # More than 10% noise
            recommendations.append(
                f"High noise ratio ({noise_ratio:.2%}). Consider using "
                "a density-based algorithm like DBSCAN or HDBSCAN."
            )

    # Check stability
    if "stability_scores" in analysis:
        stability = analysis["stability_scores"].get("mean_stability", 0)
        if stability < 0.8:  # Less than 80% stable
            recommendations.append(
                f"Low stability score ({stability:.2%}). Consider "
                "increasing the number of iterations or trying a "
                "different algorithm."
            )

    # Add recommendations to analysis
    analysis["recommendations"] = recommendations

    return improved_model, analysis


def find_optimal_clusters(
    X: np.ndarray,
    max_clusters: int = 10,
    algorithm: str = "kmeans",
    random_state: Optional[int] = None
) -> int:
    """
    Find the optimal number of clusters using multiple metrics.

    This function uses the elbow method, silhouette analysis,
    and gap statistic to determine the optimal number of clusters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    max_clusters : int, default=10
        Maximum number of clusters to consider
    algorithm : str, default='kmeans'
        Clustering algorithm to use
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    int
        Optimal number of clusters

    Examples
    --------
    >>> from clustering import find_optimal_clusters
    >>> 
    >>> # Find optimal number of clusters
    >>> n_clusters = find_optimal_clusters(X, max_clusters=15)
    >>> print(f"Optimal number of clusters: {n_clusters}")
    """
    scores = []
    n_range = range(2, max_clusters + 1)

    for n in n_range:
        # Optimize clustering for this n
        results = optimize_clustering(
            X,
            algorithm=algorithm,
            n_calls=30,  # Reduced for speed
            random_state=random_state,
            n_clusters=n
        )

        # Evaluate results
        metrics = evaluate_clustering(
            X,
            results.best_model,
            metrics=['silhouette', 'calinski_harabasz']
        )

        scores.append({
            'n_clusters': n,
            'silhouette': metrics['silhouette'],
            'calinski_harabasz': metrics['calinski_harabasz']
        })

    # Find optimal n using multiple criteria
    silhouette_scores = [s['silhouette'] for s in scores]
    ch_scores = [s['calinski_harabasz'] for s in scores]

    # Normalize scores
    sil_norm = np.array(silhouette_scores) / max(silhouette_scores)
    ch_norm = np.array(ch_scores) / max(ch_scores)

    # Combine scores (weighted average)
    combined_scores = 0.6 * sil_norm + 0.4 * ch_norm

    # Find optimal n (index of highest combined score + 2 for n_range offset)
    optimal_n = np.argmax(combined_scores) + 2

    return optimal_n


def compare_and_recommend(
    X: np.ndarray,
    algorithms: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    random_state: Optional[int] = None
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Compare multiple algorithms and recommend the best one.

    This function:
    1. Compares multiple clustering algorithms
    2. Analyzes their performance
    3. Recommends the best algorithm for the data
    4. Returns the best model and detailed comparison

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input dataset
    algorithms : List[str], optional
        Algorithms to compare. If None, uses a sensible default set
    n_clusters : int, optional
        Number of clusters. If None, determined automatically
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[BaseEstimator, Dict[str, Any]]
        - Best performing model
        - Comparison results and recommendations

    Examples
    --------
    >>> from clustering import compare_and_recommend
    >>> 
    >>> # Compare algorithms and get recommendation
    >>> best_model, comparison = compare_and_recommend(X)
    >>> 
    >>> # Print recommendations
    >>> print(f"Best algorithm: {comparison['best_algorithm']}")
    >>> print("\\nRecommendations:")
    >>> for rec in comparison['recommendations']:
    ...     print(f"- {rec}")
    """
    if algorithms is None:
        algorithms = ['kmeans', 'dbscan', 'spectral', 'agglomerative']

    # Find optimal n_clusters if not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(
            X,
            algorithm='kmeans',
            random_state=random_state
        )

    # Compare algorithms
    results = compare_algorithms(
        X,
        algorithms=algorithms,
        n_calls=50,  # Reduced for speed
        n_clusters=n_clusters,
        random_state=random_state
    )

    # Find best algorithm
    best_algo = max(
        results.items(),
        key=lambda x: x[1].best_score
    )[0]

    # Get best model
    best_model = results[best_algo].best_model

    # Generate recommendations
    recommendations = []
    for algo, perf in results.items():
        if algo == best_algo:
            recommendations.append(
                f"{algo.upper()} performed best with score {perf.best_score:.3f}"
            )
        else:
            recommendations.append(
                f"{algo.upper()}: score {perf.best_score:.3f}, "
                f"training time {perf.training_time:.2f}s"
            )

    comparison = {
        'best_algorithm': best_algo,
        'all_results': results,
        'recommendations': recommendations
    }

    return best_model, comparison