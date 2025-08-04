"""Base interface for clustering algorithms."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from .types import ParamDict, ArrayLike


class ClusteringAlgorithm(ABC):
    """Base class for all clustering algorithms.
    
    This abstract base class defines the interface that all clustering
    algorithms must implement. It provides a standard way to:
    - Create algorithm instances
    - Sample and validate parameters
    - Create scikit-learn compatible estimators
    - Get algorithm metadata
    
    Each algorithm implementation should inherit from this class and
    implement all abstract methods.
    """
    
    # Class-level constants
    CATEGORIES: ClassVar[Dict[str, str]] = {
        "basic": "Basic Clustering",
        "density": "Density-Based",
        "hierarchical": "Hierarchical",
        "affinity": "Affinity/Similarity-Based"
    }
    
    def __init__(self, random_state: Optional[int] = None):
        """Initialize the clustering algorithm.
        
        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the algorithm."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Algorithm category.
        
        Should be one of: 'basic', 'density', 'hierarchical', 'affinity'
        """
        pass
    
    @property
    @abstractmethod
    def estimator_class(self) -> type:
        """The scikit-learn estimator class for this algorithm."""
        pass
    
    @abstractmethod
    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for the algorithm using Optuna trial.
        
        This method should use the trial object to sample parameters
        within appropriate ranges for the algorithm.
        
        Parameters
        ----------
        trial : optuna.Trial
            The trial object for parameter sampling
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of sampled parameters
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for the algorithm.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of default parameters
        """
        pass
    
    def prepare_parameters(self, params: ParamDict, X: ArrayLike) -> ParamDict:
        """Prepare parameters for the clustering algorithm.
        
        This method allows algorithms to modify parameters based on the
        input data before creating the estimator. The default implementation
        returns the parameters unchanged.
        
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
        return params.copy()
    
    def create_estimator(self, params: ParamDict) -> BaseEstimator:
        """Create a new instance of the clustering estimator.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters for the estimator
        
        Returns
        -------
        BaseEstimator
            The clustering estimator instance
        """
        return self.estimator_class(**params, random_state=self.random_state)
    
    def supports_predict(self) -> bool:
        """Whether the algorithm supports predict() on new data.
        
        Most algorithms support predict(), but some (like DBSCAN)
        don't support predicting cluster labels for new data points.
        
        Returns
        -------
        bool
            True if the algorithm supports predict(), False otherwise
        """
        return True
    
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for the algorithm.
        
        This method should check that all required parameters are present
        and have valid values. It should raise ValueError for invalid parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
        
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        pass  # Default implementation accepts all parameters
    
    def __str__(self) -> str:
        """String representation of the algorithm.
        
        Returns
        -------
        str
            Algorithm name and category
        """
        return f"{self.name} ({self.CATEGORIES[self.category]})"
    
    def __repr__(self) -> str:
        """Detailed string representation.
        
        Returns
        -------
        str
            Detailed algorithm information
        """
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}', random_state={self.random_state})"