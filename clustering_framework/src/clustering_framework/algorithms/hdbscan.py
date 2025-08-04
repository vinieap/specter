"""HDBSCAN clustering algorithm implementation."""
from typing import Dict, Any

import hdbscan

from .base import BaseClusteringAlgorithm


class HDBSCANClusteringAlgorithm(BaseClusteringAlgorithm):
    """HDBSCAN clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "hdbscan"

    @property
    def estimator_class(self) -> type:
        return hdbscan.HDBSCAN

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for HDBSCAN clustering."""
        return {
            "min_cluster_size": trial.suggest_int("min_cluster_size", 2, 20),
            "min_samples": trial.suggest_int("min_samples", 1, 10),
            "cluster_selection_epsilon": trial.suggest_float(
                "cluster_selection_epsilon", 0.0, 1.0
            ),
            "alpha": trial.suggest_float("alpha", 0.5, 1.5),
            "cluster_selection_method": trial.suggest_categorical(
                "cluster_selection_method", ["eom", "leaf"]
            ),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["best", "generic", "prims_kdtree", "prims_balltree", "boruvka_kdtree", "boruvka_balltree"]
            ),
        }

    def supports_predict(self) -> bool:
        """HDBSCAN has a different prediction mechanism."""
        return False

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for HDBSCAN clustering."""
        return {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "alpha": 1.0,
            "cluster_selection_method": "eom",
            "metric": "euclidean",
            "algorithm": "best",
        }