"""Birch clustering algorithm implementation."""
from typing import Dict, Any

from sklearn.cluster import Birch

from .base import BaseClusteringAlgorithm


class BirchClusteringAlgorithm(BaseClusteringAlgorithm):
    """Birch clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "birch"

    @property
    def estimator_class(self) -> type:
        return Birch

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for Birch clustering."""
        return {
            "threshold": trial.suggest_float("threshold", 0.1, 1.0),
            "branching_factor": trial.suggest_int("branching_factor", 10, 100),
            "n_clusters": trial.suggest_int("n_clusters", 2, 18),
            "compute_labels": True,  # Always compute labels for evaluation
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Birch clustering."""
        return {
            "threshold": 0.5,
            "branching_factor": 50,
            "n_clusters": 3,
            "compute_labels": True,
        }