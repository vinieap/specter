"""Bisecting KMeans implementation."""
from typing import Dict, Any

from sklearn.cluster import BisectingKMeans

from .base import BaseKMeans
from ...core.types import ParamDict


class BisectingKMeansAlgorithm(BaseKMeans):
    """Bisecting KMeans clustering algorithm.
    
    This variant uses a divisive hierarchical strategy, recursively splitting
    clusters using standard k-means with k=2. This can produce more balanced
    clusters than standard k-means.
    """
    
    @property
    def name(self) -> str:
        return "bisecting_kmeans"
    
    @property
    def estimator_class(self) -> type:
        return BisectingKMeans
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Bisecting KMeans.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters including bisecting specific ones:
            - bisecting_strategy: 'biggest_inertia'
        """
        params = super().get_default_parameters()
        params.update({
            "bisecting_strategy": "biggest_inertia"
        })
        return params
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Bisecting KMeans.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters including bisecting specific ones:
            - bisecting_strategy: one of ['biggest_inertia', 'largest_cluster']
        """
        params = super().sample_parameters(trial)
        params.update({
            "bisecting_strategy": trial.suggest_categorical(
                "bisecting_strategy", 
                ["biggest_inertia", "largest_cluster"]
            )
        })
        return params
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Bisecting KMeans.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        super().validate_parameters(params)
        
        if "bisecting_strategy" not in params:
            raise ValueError("Missing required parameter: bisecting_strategy")
            
        if params["bisecting_strategy"] not in ["biggest_inertia", "largest_cluster"]:
            raise ValueError(
                "bisecting_strategy must be one of: 'biggest_inertia', 'largest_cluster'"
            )