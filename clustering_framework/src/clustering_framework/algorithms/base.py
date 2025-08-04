"""Base class for clustering algorithms."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class BaseClusteringAlgorithm(ABC):
    """Base class for all clustering algorithms in the framework."""

    def __init__(self, random_state: int = 42):
        """Initialize the clustering algorithm.

        Parameters
        ----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the clustering algorithm."""
        pass

    @property
    @abstractmethod
    def estimator_class(self) -> type:
        """The scikit-learn estimator class for this algorithm."""
        pass

    @abstractmethod
    def sample_parameters(self, trial) -> Dict[str, Any]:
        """Sample parameters for the algorithm using Optuna trial.

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

    def prepare_parameters(self, params: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Prepare parameters for the clustering algorithm.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters sampled by Optuna
        X : np.ndarray
            Input data to cluster

        Returns
        -------
        Dict[str, Any]
            Cleaned parameters ready for the estimator
        """
        # By default, just return the parameters as is
        # Override this method if the algorithm needs special parameter handling
        return params.copy()

    def create_estimator(self, params: Dict[str, Any]) -> BaseEstimator:
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

        Returns
        -------
        bool
            True if the algorithm supports predict(), False otherwise
        """
        # Most algorithms support predict, override if not
        return True

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the algorithm.

        Returns
        -------
        Dict[str, Any]
            Dictionary of default parameters
        """
        # Return empty dict by default, override if needed
        return {}