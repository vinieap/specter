"""
Clustering evaluation metrics.
"""

from typing import List, Dict

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metrics: List[str]
) -> Dict[str, float]:
    """
    Compute requested clustering metrics.
    
    Args:
        X: Input data matrix
        labels: Cluster labels
        metrics: List of metric names to compute
        
    Returns:
        Dictionary mapping metric names to scores
    """
    available_metrics = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score
    }
    
    results = {}
    for metric in metrics:
        if metric not in available_metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Skip if only one cluster (metrics would fail)
        if len(np.unique(labels)) < 2:
            results[metric] = 0.0
            continue
            
        score_func = available_metrics[metric]
        results[metric] = score_func(X, labels)
    
    return results