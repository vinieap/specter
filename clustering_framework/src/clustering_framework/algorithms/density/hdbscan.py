"""HDBSCAN clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from hdbscan import HDBSCAN

from .base import DensityBasedAlgorithm
from ...core.types import ParamDict


class HDBSCANAlgorithm(DensityBasedAlgorithm):
    """HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
    
    HDBSCAN extends DBSCAN by converting it into a hierarchical clustering algorithm,
    and then using a technique to extract a flat clustering based on the stability of clusters.
    """
    
    @property
    def name(self) -> str:
        return "hdbscan"
    
    @property
    def estimator_class(self) -> type:
        return HDBSCAN
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for HDBSCAN.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - min_cluster_size: 5
            - min_samples: 5
            - cluster_selection_epsilon: 0.0
            - metric: 'euclidean'
            - cluster_selection_method: 'eom'
            - leaf_size: 40
        """
        return {
            "min_cluster_size": 5,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "leaf_size": 40
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for HDBSCAN.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - min_cluster_size: int in [2, 15]
            - min_samples: int in [2, 10]
            - cluster_selection_epsilon: float in [0.0, 0.5]
            - metric: one of ['euclidean', 'manhattan', 'cosine']
            - cluster_selection_method: one of ['eom', 'leaf']
            - leaf_size: int in [20, 60]
        """
        return {
            "min_cluster_size": trial.suggest_int("min_cluster_size", 2, 15),
            "min_samples": trial.suggest_int("min_samples", 2, 10),
            "cluster_selection_epsilon": trial.suggest_float(
                "cluster_selection_epsilon", 0.0, 0.5
            ),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
            "cluster_selection_method": trial.suggest_categorical(
                "cluster_selection_method", ["eom", "leaf"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 20, 60)
        }
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for HDBSCAN.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required = {
            "min_cluster_size", "min_samples", "cluster_selection_epsilon",
            "metric", "cluster_selection_method", "leaf_size"
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["min_cluster_size"], int) or params["min_cluster_size"] < 2:
            raise ValueError("min_cluster_size must be an integer >= 2")
            
        if not isinstance(params["min_samples"], int) or params["min_samples"] < 1:
            raise ValueError("min_samples must be an integer >= 1")
            
        if not isinstance(params["cluster_selection_epsilon"], (int, float)) or params["cluster_selection_epsilon"] < 0:
            raise ValueError("cluster_selection_epsilon must be a non-negative number")
            
        if params["metric"] not in ["euclidean", "manhattan", "cosine"]:
            raise ValueError(
                "metric must be one of: 'euclidean', 'manhattan', 'cosine'"
            )
            
        if params["cluster_selection_method"] not in ["eom", "leaf"]:
            raise ValueError(
                "cluster_selection_method must be one of: 'eom', 'leaf'"
            )
            
        if not isinstance(params["leaf_size"], int) or params["leaf_size"] < 1:
            raise ValueError("leaf_size must be an integer >= 1")
    
    def get_min_samples_range(self, n_samples: int) -> tuple[int, int]:
        """Get valid range for min_samples parameter.
        
        For HDBSCAN, min_samples should be at least 1 and at most
        15% of the dataset size.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in dataset
            
        Returns
        -------
        tuple[int, int]
            Minimum and maximum values for min_samples
        """
        return (1, max(1, min(15, int(0.15 * n_samples))))