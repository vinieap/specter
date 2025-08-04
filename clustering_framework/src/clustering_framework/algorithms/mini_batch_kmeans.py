"""Mini-Batch K-Means clustering algorithm implementation."""
from typing import Dict, Any

from sklearn.cluster import MiniBatchKMeans

from .base import BaseClusteringAlgorithm


class MiniBatchKMeansClusteringAlgorithm(BaseClusteringAlgorithm):
    """Mini-Batch K-Means clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "mini_batch_kmeans"

    @property
    def estimator_class(self) -> type:
        return MiniBatchKMeans

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for Mini-Batch K-Means clustering."""
        return {
            "n_clusters": trial.suggest_int("n_clusters", 2, 18),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "batch_size": trial.suggest_int("batch_size", 100, 2000),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
            "max_no_improvement": trial.suggest_int("max_no_improvement", 5, 20),
            "reassignment_ratio": trial.suggest_float("reassignment_ratio", 0.01, 0.1),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Mini-Batch K-Means clustering."""
        return {
            "n_clusters": 8,
            "init": "k-means++",
            "max_iter": 300,
            "batch_size": 1000,
            "tol": 1e-4,
            "max_no_improvement": 10,
            "reassignment_ratio": 0.01,
        }