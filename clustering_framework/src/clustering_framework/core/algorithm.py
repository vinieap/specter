"""
Core algorithm functionality.
"""

from typing import Type, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering


ALGORITHMS: Dict[str, Type[BaseEstimator]] = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "spectral": SpectralClustering
}


def get_algorithm(name: str) -> Type[BaseEstimator]:
    """
    Get clustering algorithm class by name.
    
    Args:
        name: Name of the algorithm
        
    Returns:
        Clustering algorithm class
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {name}. Available algorithms: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[name]


class ClusteringAlgorithm:
    """Base class for clustering algorithms."""
    
    def __init__(self, algorithm: str, **kwargs: Any):
        """
        Initialize clustering algorithm.
        
        Args:
            algorithm: Name of the algorithm to use
            **kwargs: Additional parameters for the algorithm
        """
        self.algorithm = get_algorithm(algorithm)
        self.params = kwargs
        self.model = None
        
    def fit(self, X):
        """
        Fit the clustering model.
        
        Args:
            X: Input data matrix
            
        Returns:
            Fitted model
        """
        self.model = self.algorithm(**self.params)
        self.model.fit(X)
        return self.model
        
    def predict(self, X):
        """
        Predict cluster labels.
        
        Args:
            X: Input data matrix
            
        Returns:
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)