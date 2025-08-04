"""OPTICS clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import OPTICS

from .base import BaseClusteringAlgorithm


class OPTICSClusteringAlgorithm(BaseClusteringAlgorithm):
    """OPTICS clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "optics"

    @property
    def estimator_class(self) -> type:
        return OPTICS

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for OPTICS clustering."""
        return {
            "min_samples": trial.suggest_int("min_samples", 2, 20),
            "max_eps": trial.suggest_float("max_eps", 0.1, 5.0, log=True),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
            "cluster_method": trial.suggest_categorical(
                "cluster_method", ["xi", "dbscan"]
            ),
            "eps": trial.suggest_float("eps", 0.1, 5.0, log=True),
            "xi": trial.suggest_float("xi", 0.01, 0.3),
            "min_cluster_size": trial.suggest_float(
                "min_cluster_size", 0.01, 0.5
            ),  # As a fraction of min_samples
        }

    def prepare_parameters(self, params: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Prepare parameters for OPTICS clustering."""
        params_clean = params.copy()
        
        # Convert min_cluster_size from fraction to absolute number
        min_cluster_size_fraction = params_clean.pop("min_cluster_size", 0.1)
        params_clean["min_cluster_size"] = max(
            2, int(min_cluster_size_fraction * params_clean["min_samples"])
        )
        
        return params_clean

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for OPTICS clustering."""
        return {
            "min_samples": 5,
            "max_eps": float("inf"),  # Using float("inf") instead of np.inf
            "metric": "euclidean",
            "cluster_method": "xi",
            "eps": None,
            "xi": 0.05,
            "min_cluster_size": 0.1,  # As a fraction of min_samples
        }