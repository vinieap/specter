"""DBSCAN clustering algorithm implementation."""
from typing import Dict, Any

from sklearn.cluster import DBSCAN

from .base import BaseClusteringAlgorithm


class DBSCANClusteringAlgorithm(BaseClusteringAlgorithm):
    """DBSCAN clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "dbscan"

    @property
    def estimator_class(self) -> type:
        return DBSCAN

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for DBSCAN clustering."""
        return {
            "eps": trial.suggest_float("eps", 0.01, 2.0, log=True),
            "min_samples": trial.suggest_int("min_samples", 2, 10),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
        }

    def supports_predict(self) -> bool:
        """DBSCAN does not support predict() on new data."""
        return False

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for DBSCAN clustering."""
        return {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
        }