"""Affinity Propagation clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import AffinityPropagation

from .base import BaseClusteringAlgorithm


class AffinityPropagationClusteringAlgorithm(BaseClusteringAlgorithm):
    """Affinity Propagation clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "affinity_propagation"

    @property
    def estimator_class(self) -> type:
        return AffinityPropagation

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for Affinity Propagation clustering."""
        return {
            "damping": trial.suggest_float("damping", 0.5, 0.99),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "convergence_iter": trial.suggest_int("convergence_iter", 10, 30),
            "preference": trial.suggest_float("preference", -100.0, 0.0),
            "affinity": trial.suggest_categorical(
                "affinity", ["euclidean", "precomputed"]
            ),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Affinity Propagation clustering."""
        return {
            "damping": 0.5,
            "max_iter": 200,
            "convergence_iter": 15,
            "affinity": "euclidean",
        }

    def prepare_parameters(self, params: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Prepare parameters for Affinity Propagation clustering."""
        # If affinity is precomputed, we need to compute the similarity matrix
        if params.get("affinity") == "precomputed":
            from sklearn.metrics.pairwise import euclidean_distances
            # Convert distances to similarities
            similarities = -euclidean_distances(X)
            # Set the preference parameter based on similarities
            if "preference" in params:
                preference = params["preference"]
                params_clean = params.copy()
                params_clean["preference"] = float(preference * np.median(similarities))
                return params_clean
        return params.copy()