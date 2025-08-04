"""Mini-batch KMeans implementation."""
from typing import Dict, Any

import numpy as np

from sklearn.cluster import MiniBatchKMeans

from .base import BaseKMeans
from ...core.types import ParamDict


class MiniBatchKMeansAlgorithm(BaseKMeans):
    """Mini-batch KMeans clustering algorithm.
    
    This variant uses mini-batches to reduce computation time, which is useful
    for large datasets. It trades off accuracy for speed.
    """
    
    @property
    def name(self) -> str:
        return "mini_batch_kmeans"
    
    @property
    def estimator_class(self) -> type:
        return MiniBatchKMeans
    
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Mini-batch KMeans.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters including mini-batch specific ones:
            - batch_size: 1024
            - max_no_improvement: 10
        """
        params = super().get_default_parameters()
        params.update({
            "batch_size": 1024,
            "max_no_improvement": 10
        })
        return params
    
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Mini-batch KMeans.
        
        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling
            
        Returns
        -------
        Dict[str, Any]
            Sampled parameters including mini-batch specific ones:
            - batch_size: int in [256, 4096]
            - max_no_improvement: int in [5, 20]
        """
        params = super().sample_parameters(trial)
        params.update({
            "batch_size": trial.suggest_int("batch_size", 256, 4096),
            "max_no_improvement": trial.suggest_int("max_no_improvement", 5, 20)
        })
        return params
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Mini-batch KMeans.
        
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
        
        if "batch_size" not in params:
            raise ValueError("Missing required parameter: batch_size")
            
        if "max_no_improvement" not in params:
            raise ValueError("Missing required parameter: max_no_improvement")
            
        if not isinstance(params["batch_size"], int) or params["batch_size"] < 1:
            raise ValueError("batch_size must be an integer >= 1")
            
        if not isinstance(params["max_no_improvement"], int) or params["max_no_improvement"] < 0:
            raise ValueError("max_no_improvement must be an integer >= 0")
    
    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters for Mini-batch KMeans.
        
        Ensures batch_size is not larger than the dataset size.
        
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
        prepared = super().prepare_parameters(params, X)
        n_samples = X.shape[0]
        
        if prepared["batch_size"] > n_samples:
            prepared["batch_size"] = n_samples
            
        return prepared