"""Base interfaces for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type

import numpy as np
import optuna
from sklearn.base import BaseEstimator

ParamDict = Dict[str, Any]


class BaseClusteringAlgorithm(ABC):
    """Base interface for all clustering algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get algorithm name."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Get algorithm category."""
        pass

    @property
    @abstractmethod
    def estimator_class(self) -> Type[BaseEstimator]:
        """Get scikit-learn estimator class."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for the algorithm."""
        pass

    @abstractmethod
    def sample_parameters(self, trial: optuna.Trial) -> ParamDict:
        """Sample parameters using Optuna trial."""
        pass

    @abstractmethod
    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for the algorithm."""
        pass

    @abstractmethod
    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters based on input data."""
        pass
