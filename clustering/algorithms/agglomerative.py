"""Agglomerative clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .base import BaseClusteringAlgorithm


class AgglomerativeClusteringAlgorithm(BaseClusteringAlgorithm):
    """Agglomerative clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "agglomerative"

    @property
    def estimator_class(self) -> type:
        return AgglomerativeClustering

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for Agglomerative clustering."""
        params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 18),
            "linkage": trial.suggest_categorical(
                "linkage", ["ward", "complete", "average", "single"]
            ),
            "compute_distances": trial.suggest_categorical("compute_distances", [True, False]),
        }
        
        # Only suggest affinity if linkage is not 'ward'
        if params["linkage"] != "ward":
            params["affinity"] = trial.suggest_categorical(
                "affinity", ["euclidean", "manhattan", "cosine", "l1", "l2"]
            )
        else:
            # Ward linkage only works with euclidean distance
            params["affinity"] = "euclidean"
            
        # Distance threshold is an alternative to n_clusters
        if trial.suggest_categorical("use_distance_threshold", [True, False]):
            del params["n_clusters"]
            params["distance_threshold"] = trial.suggest_float(
                "distance_threshold", 0.1, 5.0, log=True
            )
            
        return params

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Agglomerative clustering."""
        return {
            "n_clusters": 2,
            "affinity": "euclidean",
            "linkage": "ward",
            "compute_distances": True,
        }