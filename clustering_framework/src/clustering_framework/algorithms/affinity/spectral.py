"""Spectral clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import SpectralClustering

from .base import AffinityBasedAlgorithm
from ...core.types import ParamDict


class SpectralClusteringAlgorithm(AffinityBasedAlgorithm):
    """Spectral Clustering.
    
    This algorithm performs dimensionality reduction using the eigenvalues of a
    similarity matrix, then applies k-means in the reduced space. It's particularly
    good at finding non-spherical clusters.
    """
    
    @property
    def name(self) -> str:
        return "spectral"
    
    @property
    def estimator_class(self) -> type:
        return SpectralClustering
    
    def requires_precomputed_affinity(self) -> bool:
        return False
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Spectral Clustering.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - n_clusters: 8
            - eigen_solver: None (auto)
            - n_components: None (n_clusters)
            - random_state: None
            - n_init: 10
            - gamma: 1.0
            - affinity: 'rbf'
            - n_neighbors: 10
            - eigen_tol: 0.0
            - assign_labels: 'kmeans'
            - degree: 3
            - coef0: 1
            - kernel_params: None
            - n_jobs: None
        """
        return {
            "n_clusters": 8,
            "eigen_solver": None,
            "n_components": None,
            "n_init": 10,
            "gamma": 1.0,
            "affinity": "rbf",
            "n_neighbors": 10,
            "eigen_tol": 0.0,
            "assign_labels": "kmeans",
            "degree": 3,
            "coef0": 1,
            "kernel_params": None,
            "n_jobs": None
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Spectral Clustering.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - n_clusters: int in [2, 20]
            - n_init: int in [5, 20]
            - gamma: float in [0.1, 10.0]
            - affinity: one of ['rbf', 'nearest_neighbors', 'polynomial']
            - n_neighbors: int in [5, 20]
            - assign_labels: one of ['kmeans', 'discretize']
            - degree: int in [2, 5]
            - coef0: float in [0.0, 2.0]
        """
        params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "n_init": trial.suggest_int("n_init", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.1, 10.0, log=True),
            "affinity": trial.suggest_categorical(
                "affinity", ["rbf", "nearest_neighbors", "polynomial"]
            ),
            "assign_labels": trial.suggest_categorical(
                "assign_labels", ["kmeans", "discretize"]
            )
        }
        
        if params["affinity"] == "nearest_neighbors":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 5, 20)
        elif params["affinity"] == "polynomial":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 2.0)
            
        return params
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Spectral Clustering.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required = {"n_clusters", "n_init", "gamma", "affinity", "assign_labels"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["n_clusters"], int) or params["n_clusters"] < 2:
            raise ValueError("n_clusters must be an integer >= 2")
            
        if not isinstance(params["n_init"], int) or params["n_init"] < 1:
            raise ValueError("n_init must be an integer >= 1")
            
        if not isinstance(params["gamma"], (int, float)) or params["gamma"] <= 0:
            raise ValueError("gamma must be a positive number")
            
        valid_affinities = ["rbf", "nearest_neighbors", "polynomial"]
        if params["affinity"] not in valid_affinities:
            raise ValueError(
                f"affinity must be one of: {', '.join(valid_affinities)}"
            )
            
        if params["assign_labels"] not in ["kmeans", "discretize"]:
            raise ValueError(
                "assign_labels must be one of: 'kmeans', 'discretize'"
            )
            
        if params["affinity"] == "nearest_neighbors":
            if "n_neighbors" not in params:
                raise ValueError(
                    "n_neighbors is required when affinity='nearest_neighbors'"
                )
            if not isinstance(params["n_neighbors"], int) or params["n_neighbors"] < 2:
                raise ValueError("n_neighbors must be an integer >= 2")
                
        elif params["affinity"] == "polynomial":
            if "degree" not in params:
                raise ValueError(
                    "degree is required when affinity='polynomial'"
                )
            if not isinstance(params["degree"], int) or params["degree"] < 1:
                raise ValueError("degree must be an integer >= 1")
                
            if "coef0" not in params:
                raise ValueError(
                    "coef0 is required when affinity='polynomial'"
                )
            if not isinstance(params["coef0"], (int, float)):
                raise ValueError("coef0 must be a number")