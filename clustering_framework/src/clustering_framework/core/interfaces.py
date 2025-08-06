"""Interface definitions for component boundaries."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, runtime_checkable

import numpy as np
import optuna


@runtime_checkable
class ParameterProvider(Protocol):
    """Protocol for components that provide parameter management."""

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters."""
        ...

    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate parameters."""
        ...

    def prepare_parameters(
        self, params: Dict[str, Any], X: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare parameters based on input data."""
        ...


@runtime_checkable
class OptimizationProvider(Protocol):
    """Protocol for components that provide optimization capabilities."""

    def sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters for optimization."""
        ...

    def evaluate(self, X: np.ndarray, params: Dict[str, Any]) -> float:
        """Evaluate parameters on input data."""
        ...


class AnalysisComponent(ABC):
    """Base class for analysis components."""

    @abstractmethod
    def analyze(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering results.

        Parameters
        ----------
        X : array-like
            Input data
        labels : array-like
            Cluster labels

        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        pass


class MetricsProvider(ABC):
    """Base class for components that provide metrics."""

    @abstractmethod
    def calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering metrics.

        Parameters
        ----------
        X : array-like
            Input data
        labels : array-like
            Cluster labels

        Returns
        -------
        Dict[str, float]
            Metric values
        """
        pass


class ProcessManager(ABC):
    """Base class for process management components."""

    @abstractmethod
    def start(self) -> None:
        """Start the managed process."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the managed process."""
        pass

    @abstractmethod
    def restart(self) -> None:
        """Restart the managed process."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get process status."""
        pass


class ResourceManager(ABC):
    """Base class for resource management components."""

    @abstractmethod
    def acquire(self) -> Any:
        """Acquire a resource."""
        pass

    @abstractmethod
    def release(self, resource: Any) -> None:
        """Release a resource."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up all resources."""
        pass
