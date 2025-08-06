"""Base interface for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, ClassVar

import numpy as np
import optuna
from sklearn.base import BaseEstimator

ParamDict = Dict[str, Any]


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
        "affinity": "Affinity/Similarity-Based",
    }

    def __init__(self, random_state: Optional[int] = None):
        """Initialize the clustering algorithm.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state if self.supports_random_state else None
        self._parameters = self.get_default_parameters()

    @property
    def supports_random_state(self) -> bool:
        """Whether this algorithm supports random state initialization."""
        return hasattr(self.estimator_class(), "random_state")

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
    def estimator_class(self) -> Type[BaseEstimator]:
        """The scikit-learn estimator class for this algorithm."""
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

    @abstractmethod
    def sample_parameters(self, trial: optuna.Trial) -> ParamDict:
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
        pass

    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters based on input data.

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

    def create_estimator(self, params: Optional[ParamDict] = None) -> BaseEstimator:
        """Create a new instance of the clustering estimator.

        Parameters
        ----------
        params : Dict[str, Any], optional
            Additional parameters for the estimator. If provided, these will be
            merged with the stored parameters.

        Returns
        -------
        BaseEstimator
            The clustering estimator instance
        """
        estimator_params = self._parameters.copy()
        if params:
            estimator_params.update(params)
        if self.supports_random_state:
            estimator_params["random_state"] = self.random_state

        # Filter parameters to only include those accepted by the estimator
        estimator = self.estimator_class()
        valid_params = {
            k: v
            for k, v in estimator_params.items()
            if hasattr(estimator, k) or k in estimator.get_params()
        }

        return self.estimator_class(**valid_params)

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

    def set_parameters(self, params: ParamDict) -> None:
        """Set algorithm parameters.

        This method validates and sets the parameters for the algorithm.
        It merges the provided parameters with the existing ones.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to set

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # Merge with existing parameters
        new_params = self._parameters.copy()
        new_params.update(params)

        # Validate parameters
        self.validate_parameters(new_params)

        # Update parameters
        self._parameters = new_params
