"""Type definitions for the clustering framework.

This module contains type definitions, protocols, and type aliases used throughout
the framework to ensure type safety and provide better IDE support.
"""

from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
    Sequence,
    Tuple,
    Callable,
)

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin

# Type variables
T = TypeVar("T")
EstimatorT = TypeVar("EstimatorT", bound=BaseEstimator)
ClusteringEstimatorT = TypeVar(
    "ClusteringEstimatorT", bound=Union[BaseEstimator, ClusterMixin]
)

# Type aliases
ParamDict = Dict[str, Any]
OptionalParamDict = Optional[ParamDict]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# Protocol definitions
@runtime_checkable
class HasFit(Protocol[EstimatorT]):
    """Protocol for objects that have a fit method."""

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any
    ) -> EstimatorT:
        """Fit the model to data."""
        ...


@runtime_checkable
class HasPredict(Protocol):
    """Protocol for objects that have a predict method."""

    def predict(self, X: ArrayLike) -> IntArray:
        """Predict cluster labels for samples in X."""
        ...


@runtime_checkable
class HasFitPredict(HasFit[EstimatorT], HasPredict, Protocol[EstimatorT]):
    """Protocol for objects that have both fit and predict methods."""

    pass


@runtime_checkable
class ClusteringEstimator(
    HasFitPredict[ClusteringEstimatorT], Protocol[ClusteringEstimatorT]
):
    """Protocol for clustering estimators."""

    n_clusters_: int
    labels_: IntArray

    def fit_predict(
        self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any
    ) -> IntArray:
        """Fit the model and predict cluster labels for X."""
        ...


# Result types
@dataclass
class OptimizationResult:
    """Results from a clustering optimization run."""

    best_score: float
    best_params: ParamDict
    best_model: ClusteringEstimator
    history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_stats: Dict[str, Any]


@dataclass
class ValidationResult:
    """Results from parameter validation."""

    is_valid: bool
    errors: List[str]
    context: Optional[Dict[str, Any]] = None


@dataclass
class NoiseAnalysis:
    """Results from noise analysis."""

    noise_indices: IntArray
    noise_scores: FloatArray
    threshold: float
    stats: Dict[str, Any]


@dataclass
class AlgorithmPerformance:
    """Performance metrics for a clustering algorithm."""

    name: str
    score: float
    metrics: Dict[str, float]
    timing: Dict[str, float]
    memory_usage: Dict[str, int]


# Callback types
MetricFunction = Callable[[ArrayLike, IntArray], float]
ProgressCallback = Callable[[int, int, Dict[str, Any]], None]
ValidationFunction = Callable[[ParamDict], ValidationResult]


# Convergence types
@dataclass
class ConvergenceStatus:
    """Status of optimization convergence detection.

    This class provides information about whether optimization has converged,
    how confident we are in the convergence, and recommendations for
    continuing or stopping optimization.

    Attributes
    ----------
    converged : bool
        Whether optimization has converged
    confidence : float
        Confidence in convergence detection (0 to 1)
    method : str
        Method used to detect convergence
    details : Dict[str, Any]
        Detailed convergence metrics
    recommendation : str
        Recommendation for optimization process
    """

    converged: bool
    confidence: float
    method: str
    details: Dict[str, Any]
    recommendation: str


# Configuration types
@dataclass
class AlgorithmConfig:
    """Base configuration for clustering algorithms."""

    name: str
    category: str
    params: ParamDict
    param_ranges: Dict[str, Tuple[float, float]]
    param_choices: Dict[str, List[Any]]
    param_dependencies: Dict[str, Dict[str, List[Any]]]

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.category:
            raise ValueError("category must not be empty")
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary")
        if not isinstance(self.param_ranges, dict):
            raise ValueError("param_ranges must be a dictionary")
        if not isinstance(self.param_choices, dict):
            raise ValueError("param_choices must be a dictionary")
        if not isinstance(self.param_dependencies, dict):
            raise ValueError("param_dependencies must be a dictionary")
