"""
Custom exceptions for the clustering framework.

This module defines a hierarchy of exceptions specific to the clustering framework,
allowing for precise error handling and reporting throughout the system.
"""

from typing import Any, Dict, Optional


class ClusteringError(Exception):
    """Base exception class for all clustering framework errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the exception with a message and optional context.

        Args:
            message: Human-readable error description
            context: Optional dictionary containing additional error context
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ConfigurationError(ClusteringError):
    """Raised when there is an error in the configuration settings."""

    pass


class ValidationError(ClusteringError):
    """Raised when validation fails for parameters or data."""

    pass


class AlgorithmError(ClusteringError):
    """Base class for algorithm-specific errors."""

    pass


class InitializationError(AlgorithmError):
    """Raised when algorithm initialization fails."""

    pass


class ConvergenceError(AlgorithmError):
    """Raised when an algorithm fails to converge."""

    pass


class OptimizationError(ClusteringError):
    """Base class for optimization-related errors."""

    pass


class ParameterError(OptimizationError):
    """Raised when there are issues with parameter values or ranges."""

    pass


class ResourceError(ClusteringError):
    """Base class for resource-related errors."""

    pass


class MemoryError(ResourceError):
    """Raised when memory-related issues occur."""

    pass


class ComputationError(ResourceError):
    """Raised when computational resources are exhausted or unavailable."""

    pass


class DataError(ClusteringError):
    """Base class for data-related errors."""

    pass


class DataTypeError(DataError):
    """Raised when data has incorrect type."""

    pass


class DataShapeError(DataError):
    """Raised when data has incorrect shape or dimensions."""

    pass


class DataQualityError(DataError):
    """Raised when data quality issues are detected."""

    pass


class MetricError(ClusteringError):
    """Base class for metric-related errors."""

    pass


class MetricComputationError(MetricError):
    """Raised when metric computation fails."""

    pass


class MetricValidationError(MetricError):
    """Raised when metric validation fails."""

    pass


class ParallelizationError(ClusteringError):
    """Base class for parallelization-related errors."""

    pass


class ProcessError(ParallelizationError):
    """Raised when process-related issues occur."""

    pass


class CommunicationError(ParallelizationError):
    """Raised when inter-process communication fails."""

    pass


class PersistenceError(ClusteringError):
    """Base class for data persistence errors."""

    pass


class SerializationError(PersistenceError):
    """Raised when serialization/deserialization fails."""

    pass


class StorageError(PersistenceError):
    """Raised when storage operations fail."""

    pass


class ExternalError(ClusteringError):
    """Base class for errors in external dependencies or integrations."""

    pass


class DependencyError(ExternalError):
    """Raised when there are issues with external dependencies."""

    pass


class IntegrationError(ExternalError):
    """Raised when integration with external systems fails."""

    pass
