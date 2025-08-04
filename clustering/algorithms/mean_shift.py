"""Mean Shift clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from .base import BaseClusteringAlgorithm


class MeanShiftClusteringAlgorithm(BaseClusteringAlgorithm):
    """Mean Shift clustering algorithm with Optuna optimization support."""

    @property
    def name(self) -> str:
        return "mean_shift"

    @property
    def estimator_class(self) -> type:
        return MeanShift

    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for Mean Shift clustering."""
        return {
            "bandwidth_scale": trial.suggest_float("bandwidth_scale", 0.1, 2.0, log=True),
            "bin_seeding": trial.suggest_categorical("bin_seeding", [True, False]),
            "cluster_all": trial.suggest_categorical("cluster_all", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
        }

    def prepare_parameters(self, params: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Prepare parameters for Mean Shift clustering."""
        params_clean = params.copy()
        
        # Estimate bandwidth using the scale factor
        bandwidth_scale = params_clean.pop("bandwidth_scale", 1.0)
        bandwidth = estimate_bandwidth(X, quantile=0.3) * bandwidth_scale
        params_clean["bandwidth"] = bandwidth
        
        return params_clean

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Mean Shift clustering."""
        return {
            "bandwidth_scale": 1.0,
            "bin_seeding": False,
            "cluster_all": True,
            "max_iter": 300,
        }