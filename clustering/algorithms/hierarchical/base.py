"""Base class for hierarchical clustering algorithms."""
from abc import abstractmethod
from typing import Dict, Any

import numpy as np

from ...core.algorithm import ClusteringAlgorithm
from ...core.types import ParamDict


class HierarchicalAlgorithm(ClusteringAlgorithm):
    """Base class for hierarchical clustering algorithms.
    
    Hierarchical clustering algorithms create a tree of clusters by either
    merging smaller clusters (agglomerative) or splitting larger ones (divisive).
    This base class provides common functionality for both approaches.
    """
    
    @property
    def category(self) -> str:
        return "hierarchical"
    
    @abstractmethod
    def is_agglomerative(self) -> bool:
        """Whether this is an agglomerative (bottom-up) algorithm.
        
        Returns
        -------
        bool
            True for agglomerative, False for divisive
        """
        pass
    
    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters based on input data.
        
        For hierarchical algorithms, this mainly involves adjusting the
        number of clusters based on dataset size.
        
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
        
        if "n_clusters" in prepared:
            prepared["n_clusters"] = min(prepared["n_clusters"], n_samples)
            
        return prepared