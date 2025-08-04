"""Affinity Propagation clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import AffinityPropagation

from .base import AffinityBasedAlgorithm
from ...core.types import ParamDict


class AffinityPropagationAlgorithm(AffinityBasedAlgorithm):
    """Affinity Propagation Clustering.
    
    This algorithm creates clusters by passing messages between pairs of samples
    until convergence. Each message represents the suitability of one sample
    to serve as the exemplar for another sample.
    """
    
    @property
    def name(self) -> str:
        return "affinity_propagation"
    
    @property
    def estimator_class(self) -> type:
        return AffinityPropagation
    
    def requires_precomputed_affinity(self) -> bool:
        return False
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Affinity Propagation.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - damping: 0.5
            - max_iter: 200
            - convergence_iter: 15
            - copy: True
            - preference: None
            - affinity: 'euclidean'
            - verbose: False
        """
        return {
            "damping": 0.5,
            "max_iter": 200,
            "convergence_iter": 15,
            "copy": True,
            "preference": None,
            "affinity": "euclidean",
            "verbose": False
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Affinity Propagation.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - damping: float in [0.5, 0.9]
            - max_iter: int in [100, 500]
            - convergence_iter: int in [10, 30]
            - affinity: one of ['euclidean', 'precomputed']
        """
        return {
            "damping": trial.suggest_float("damping", 0.5, 0.9),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "convergence_iter": trial.suggest_int("convergence_iter", 10, 30),
            "affinity": trial.suggest_categorical(
                "affinity", ["euclidean", "precomputed"]
            ),
            "copy": True,
            "preference": None,
            "verbose": False
        }
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Affinity Propagation.
        
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
            "damping", "max_iter", "convergence_iter", "copy",
            "preference", "affinity", "verbose"
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["damping"], (int, float)) or not 0.5 <= params["damping"] <= 1.0:
            raise ValueError("damping must be a number between 0.5 and 1.0")
            
        if not isinstance(params["max_iter"], int) or params["max_iter"] < 1:
            raise ValueError("max_iter must be an integer >= 1")
            
        if not isinstance(params["convergence_iter"], int) or params["convergence_iter"] < 1:
            raise ValueError("convergence_iter must be an integer >= 1")
            
        if params["affinity"] not in ["euclidean", "precomputed"]:
            raise ValueError(
                "affinity must be one of: 'euclidean', 'precomputed'"
            )
            
        if not isinstance(params["copy"], bool):
            raise ValueError("copy must be a boolean")
            
        if not isinstance(params["verbose"], bool):
            raise ValueError("verbose must be a boolean")
            
        if params["preference"] is not None and not isinstance(params["preference"], (int, float, np.ndarray)):
            raise ValueError(
                "preference must be None or a number or array-like"
            )