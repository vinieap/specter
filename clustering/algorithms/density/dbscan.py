"""DBSCAN clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import DBSCAN

from .base import DensityBasedAlgorithm
from ...core.types import ParamDict


class DBSCANAlgorithm(DensityBasedAlgorithm):
    """DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    DBSCAN finds core samples of high density and expands clusters from them.
    It's good at finding clusters of arbitrary shape and identifying noise points.
    """
    
    @property
    def name(self) -> str:
        return "dbscan"
    
    @property
    def estimator_class(self) -> type:
        return DBSCAN
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for DBSCAN.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - eps: 0.5
            - min_samples: 5
            - metric: 'euclidean'
            - algorithm: 'auto'
            - leaf_size: 30
        """
        return {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "algorithm": "auto",
            "leaf_size": 30
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for DBSCAN.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - eps: float in [0.1, 1.0]
            - min_samples: int in [2, 10]
            - metric: one of ['euclidean', 'manhattan', 'cosine']
            - algorithm: one of ['auto', 'ball_tree', 'kd_tree', 'brute']
            - leaf_size: int in [20, 50]
        """
        return {
            "eps": trial.suggest_float("eps", 0.1, 1.0),
            "min_samples": trial.suggest_int("min_samples", 2, 10),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 20, 50)
        }
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for DBSCAN.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required = {"eps", "min_samples", "metric", "algorithm", "leaf_size"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["eps"], (int, float)) or params["eps"] <= 0:
            raise ValueError("eps must be a positive number")
            
        if not isinstance(params["min_samples"], int) or params["min_samples"] < 2:
            raise ValueError("min_samples must be an integer >= 2")
            
        if params["metric"] not in ["euclidean", "manhattan", "cosine"]:
            raise ValueError(
                "metric must be one of: 'euclidean', 'manhattan', 'cosine'"
            )
            
        if params["algorithm"] not in ["auto", "ball_tree", "kd_tree", "brute"]:
            raise ValueError(
                "algorithm must be one of: 'auto', 'ball_tree', 'kd_tree', 'brute'"
            )
            
        if not isinstance(params["leaf_size"], int) or params["leaf_size"] < 1:
            raise ValueError("leaf_size must be an integer >= 1")
    
    def get_min_samples_range(self, n_samples: int) -> tuple[int, int]:
        """Get valid range for min_samples parameter.
        
        For DBSCAN, min_samples should be at least 2 and at most
        10% of the dataset size.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in dataset
            
        Returns
        -------
        tuple[int, int]
            Minimum and maximum values for min_samples
        """
        return (2, max(2, min(10, int(0.1 * n_samples))))