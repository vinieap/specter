"""Base class for affinity/similarity-based clustering algorithms."""
from abc import abstractmethod
from typing import Dict, Any

import numpy as np

from ...core.algorithm import ClusteringAlgorithm
from ...core.types import ParamDict


class AffinityBasedAlgorithm(ClusteringAlgorithm):
    """Base class for affinity/similarity-based clustering algorithms.
    
    These algorithms use a similarity/affinity matrix to determine cluster
    assignments. They are particularly good at finding non-spherical clusters
    and can work with arbitrary similarity measures.
    """
    
    @property
    def category(self) -> str:
        return "affinity"
    
    @abstractmethod
    def requires_precomputed_affinity(self) -> bool:
        """Whether the algorithm requires a precomputed affinity matrix.
        
        Returns
        -------
        bool
            True if precomputed affinity matrix is required
        """
        pass
    
    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters based on input data.
        
        For affinity-based algorithms, this mainly involves adjusting parameters
        that depend on the dataset size or dimensionality.
        
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
        
        # Adjust n_clusters if present
        if "n_clusters" in prepared:
            prepared["n_clusters"] = min(prepared["n_clusters"], n_samples)
            
        # Adjust n_neighbors if present (for spectral clustering)
        if "n_neighbors" in prepared:
            prepared["n_neighbors"] = min(
                prepared["n_neighbors"],
                max(2, n_samples - 1)
            )
            
        return prepared