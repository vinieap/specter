"""
Core API functions for the clustering framework.
"""

from typing import Dict, List, Optional, Union, Any, Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import SpectralClustering, DBSCAN

from .optimizer import ClusteringOptimizer
from .algorithm import get_algorithm
from ..analysis.stability import analyze_stability
from ..analysis.noise import analyze_noise
from ..utils.metrics import compute_metrics


def optimize_clustering(
    X: np.ndarray,
    algorithm: str = "kmeans",
    n_calls: int = 50,
    random_state: Optional[int] = None,
    **kwargs: Any
) -> ClusteringOptimizer:
    """
    Optimize clustering parameters for the given algorithm.
    
    Args:
        X: Input data matrix
        algorithm: Name of clustering algorithm to use
        n_calls: Number of optimization iterations
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the algorithm
        
    Returns:
        Optimization results containing best model and history
    """
    # Remove random_state from kwargs if algorithm doesn't support it
    algo_class = get_algorithm(algorithm)
    if algorithm == "dbscan" and "random_state" in kwargs:
        del kwargs["random_state"]
    
    optimizer = ClusteringOptimizer(
        algorithm=algo_class,
        n_calls=n_calls,
        random_state=random_state,
        **kwargs
    )
    optimizer.fit(X)
    return optimizer


def analyze_clusters(
    X: np.ndarray,
    model: BaseEstimator,
    noise_analysis: bool = True,
    stability_analysis: bool = True
) -> Dict[str, Any]:
    """
    Analyze clustering results.
    
    Args:
        X: Input data matrix
        model: Fitted clustering model
        noise_analysis: Whether to analyze noise sensitivity
        stability_analysis: Whether to analyze cluster stability
        
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    
    if noise_analysis:
        results["noise_analysis"] = analyze_noise(X, model)
    
    if stability_analysis:
        results["stability_scores"] = analyze_stability(X, model)
    
    return results


def evaluate_clustering(
    X: np.ndarray,
    model: BaseEstimator,
    metrics: List[str] = ["silhouette", "calinski_harabasz"]
) -> Dict[str, float]:
    """
    Evaluate clustering results using various metrics.
    
    Args:
        X: Input data matrix
        model: Fitted clustering model
        metrics: List of metric names to compute
        
    Returns:
        Dictionary mapping metric names to scores
    """
    # Get labels based on model type
    if isinstance(model, (SpectralClustering, DBSCAN)):
        labels = model.labels_
    else:
        labels = model.predict(X)
        
    return compute_metrics(X, labels, metrics)


def quick_cluster(
    X: np.ndarray,
    n_clusters: Optional[int] = 3,
    algorithm: str = "kmeans",
    random_state: Optional[int] = None,
    **kwargs: Any
) -> tuple[BaseEstimator, Dict[str, float]]:
    """
    Quick clustering with basic evaluation.
    
    Args:
        X: Input data matrix
        n_clusters: Number of clusters (ignored for DBSCAN)
        algorithm: Clustering algorithm to use
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the algorithm
        
    Returns:
        Tuple of (fitted model, evaluation metrics)
    """
    algo_class = get_algorithm(algorithm)
    
    # Handle algorithm-specific parameters
    if algorithm == "dbscan":
        model = algo_class(**kwargs)
    else:
        if random_state is not None:
            kwargs["random_state"] = random_state
        if n_clusters is not None:
            kwargs["n_clusters"] = n_clusters
        model = algo_class(**kwargs)
    
    model.fit(X)
    metrics = evaluate_clustering(X, model)
    return model, metrics