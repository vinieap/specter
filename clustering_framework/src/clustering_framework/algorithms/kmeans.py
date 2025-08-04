"""K-Means clustering algorithm implementation."""
from typing import Dict, Any

from sklearn.cluster import KMeans

from .base import BaseClusteringAlgorithm


class KMeansClusteringAlgorithm(BaseClusteringAlgorithm):
    """K-Means clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "kmeans"

    @property
    def estimator_class(self) -> type:
        return KMeans

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for K-Means clustering."""
        return {
            "n_clusters": trial.suggest_int("n_clusters", 2, 18),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "n_init": trial.suggest_int("n_init", 5, 15),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for K-Means clustering."""
        return {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
        }