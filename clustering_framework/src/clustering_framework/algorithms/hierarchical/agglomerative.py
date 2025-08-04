"""Agglomerative clustering algorithm implementation."""
from typing import Dict, Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .base import HierarchicalAlgorithm
from ...core.types import ParamDict


class AgglomerativeClusteringAlgorithm(HierarchicalAlgorithm):
    """Agglomerative Hierarchical Clustering.
    
    This algorithm performs hierarchical clustering using a bottom-up approach:
    each observation starts in its own cluster, and clusters are successively
    merged together based on their similarity.
    """
    
    @property
    def name(self) -> str:
        return "agglomerative"
    
    @property
    def estimator_class(self) -> type:
        return AgglomerativeClustering
    
    def is_agglomerative(self) -> bool:
        return True
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Agglomerative Clustering.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - n_clusters: 2
            - affinity: 'euclidean'
            - linkage: 'ward'
            - compute_distances: True
        """
        return {
            "n_clusters": 2,
            "affinity": "euclidean",
            "linkage": "ward",
            "compute_distances": True
        }
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Agglomerative Clustering.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - n_clusters: int in [2, 20]
            - affinity: one of ['euclidean', 'manhattan', 'cosine']
            - linkage: one of ['ward', 'complete', 'average', 'single']
            - compute_distances: bool
        """
        params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "compute_distances": trial.suggest_categorical(
                "compute_distances", [True, False]
            )
        }
        
        # Ward linkage only works with euclidean affinity
        if trial.suggest_categorical("use_ward", [True, False]):
            params["linkage"] = "ward"
            params["affinity"] = "euclidean"
        else:
            params["affinity"] = trial.suggest_categorical(
                "affinity", ["euclidean", "manhattan", "cosine"]
            )
            params["linkage"] = trial.suggest_categorical(
                "linkage", ["complete", "average", "single"]
            )
            
        return params
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Agglomerative Clustering.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required = {"n_clusters", "affinity", "linkage", "compute_distances"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        if not isinstance(params["n_clusters"], int) or params["n_clusters"] < 2:
            raise ValueError("n_clusters must be an integer >= 2")
            
        valid_affinities = ["euclidean", "manhattan", "cosine"]
        if params["affinity"] not in valid_affinities:
            raise ValueError(
                f"affinity must be one of: {', '.join(valid_affinities)}"
            )
            
        valid_linkages = ["ward", "complete", "average", "single"]
        if params["linkage"] not in valid_linkages:
            raise ValueError(
                f"linkage must be one of: {', '.join(valid_linkages)}"
            )
            
        if params["linkage"] == "ward" and params["affinity"] != "euclidean":
            raise ValueError("Ward linkage only works with euclidean affinity")
            
        if not isinstance(params["compute_distances"], bool):
            raise ValueError("compute_distances must be a boolean")