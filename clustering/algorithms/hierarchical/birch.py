"""BIRCH clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import Birch

from .base import HierarchicalAlgorithm
from ...core.types import ParamDict


class BirchAlgorithm(HierarchicalAlgorithm):
    """BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).
    
    BIRCH builds a tree structure called CFTree to incrementally cluster the data.
    It's particularly effective for large datasets as it only needs one pass over
    the data to create a good clustering.
    """
    
    @property
    def name(self) -> str:
        return "birch"
    
    @property
    def estimator_class(self) -> type:
        return Birch
    
    def is_agglomerative(self) -> bool:
        return False  # BIRCH is divisive
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for BIRCH.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - threshold: 0.5
            - branching_factor: 50
            - n_clusters: 3
            - compute_labels: True
            - copy: True
        """
        return {
            "threshold": 0.5,
            "branching_factor": 50,
            "n_clusters": 3,
            "compute_labels": True,
            "copy": True
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for BIRCH.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - threshold: float in [0.1, 1.0]
            - branching_factor: int in [10, 100]
            - n_clusters: int in [2, 20]
            - compute_labels: bool
            - copy: bool
        """
        return {
            "threshold": trial.suggest_float("threshold", 0.1, 1.0),
            "branching_factor": trial.suggest_int("branching_factor", 10, 100),
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "compute_labels": trial.suggest_categorical(
                "compute_labels", [True, False]
            ),
            "copy": trial.suggest_categorical("copy", [True, False])
        }
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for BIRCH.
        
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
            "threshold", "branching_factor", "n_clusters",
            "compute_labels", "copy"
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["threshold"], (int, float)) or params["threshold"] <= 0:
            raise ValueError("threshold must be a positive number")
            
        if not isinstance(params["branching_factor"], int) or params["branching_factor"] < 2:
            raise ValueError("branching_factor must be an integer >= 2")
            
        if not isinstance(params["n_clusters"], int) or params["n_clusters"] < 2:
            raise ValueError("n_clusters must be an integer >= 2")
            
        if not isinstance(params["compute_labels"], bool):
            raise ValueError("compute_labels must be a boolean")
            
        if not isinstance(params["copy"], bool):
            raise ValueError("copy must be a boolean")