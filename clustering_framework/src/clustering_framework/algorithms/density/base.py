"""Base class for density-based clustering algorithms."""
from abc import abstractmethod
from typing import Dict, Any

import numpy as np

from ...core.algorithm import ClusteringAlgorithm
from ...core.types import ParamDict


class DensityBasedAlgorithm(ClusteringAlgorithm):
    """Base class for density-based clustering algorithms.
    
    Density-based algorithms find clusters by identifying areas of high density
    separated by areas of low density. This base class provides common
    functionality for parameter validation and preparation.
    """
    
    @property
    def category(self) -> str:
        return "density"
    
    def supports_predict(self) -> bool:
        """Most density-based algorithms don't support predict().
        
        Returns
        -------
        bool
            False by default for density-based algorithms
        """
        return False
    
    @abstractmethod
    def get_min_samples_range(self, n_samples: int) -> tuple[int, int]:
        """Get valid range for min_samples parameter.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in dataset
            
        Returns
        -------
        tuple[int, int]
            Minimum and maximum values for min_samples
        """
        pass
    
    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters based on input data.
        
        Ensures min_samples is not larger than the dataset size.
        
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
        
        if "min_samples" in prepared:
            min_samples_range = self.get_min_samples_range(n_samples)
            prepared["min_samples"] = max(
                min_samples_range[0],
                min(prepared["min_samples"], min_samples_range[1])
            )
            
        return prepared