"""Base KMeans implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import KMeans

from ...core.algorithm import ClusteringAlgorithm
from ...core.types import ParamDict


class BaseKMeans(ClusteringAlgorithm):
    """Base class for KMeans clustering algorithms.
    
    This class implements the core KMeans functionality and parameter validation
    that is shared across different KMeans variants.
    """
    
    @property
    def name(self) -> str:
        return "kmeans"
    
    @property
    def category(self) -> str:
        return "basic"
    
    @property
    def estimator_class(self) -> type:
        return KMeans
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for KMeans.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - n_clusters: 8
            - init: 'k-means++'
            - n_init: 10
            - max_iter: 300
            - tol: 1e-4
        """
        return {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for KMeans using Optuna trial.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - n_clusters: int in [2, 20]
            - init: one of ['k-means++', 'random']
            - n_init: int in [5, 20]
            - max_iter: int in [100, 500]
            - tol: float in [1e-5, 1e-3]
        """
        return {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "n_init": trial.suggest_int("n_init", 5, 20),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True)
        }
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for KMeans.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required_params = {"n_clusters", "init", "n_init", "max_iter", "tol"}
        missing = required_params - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["n_clusters"], int) or params["n_clusters"] < 2:
            raise ValueError("n_clusters must be an integer >= 2")
            
        if params["init"] not in ["k-means++", "random"]:
            raise ValueError("init must be 'k-means++' or 'random'")
            
        if not isinstance(params["n_init"], int) or params["n_init"] < 1:
            raise ValueError("n_init must be an integer >= 1")
            
        if not isinstance(params["max_iter"], int) or params["max_iter"] < 1:
            raise ValueError("max_iter must be an integer >= 1")
            
        if not isinstance(params["tol"], (int, float)) or params["tol"] <= 0:
            raise ValueError("tol must be a positive number")
    
    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters for KMeans based on input data.
        
        Ensures n_clusters is not larger than the number of samples.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to prepare
        X : array-like
            Input data to cluster
            
        Returns
        -------
        Dict[str, Any]
            Prepared parameters
        """
        prepared = params.copy()
        n_samples = X.shape[0]
        
        if prepared["n_clusters"] > n_samples:
            prepared["n_clusters"] = n_samples
            
        return prepared