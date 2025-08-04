"""Core components of the clustering library."""

from .algorithm import ClusteringAlgorithm
from .optimizer import ClusteringOptimizer
from .registry import algorithm_registry
from .types import (
    OptimizationResult,
    NoiseAnalysis,
    ConvergenceStatus,
    AlgorithmPerformance,
    ParamDict,
    OptionalParamDict,
    ArrayLike,
)

__all__ = [
    # Base classes
    "ClusteringAlgorithm",
    "ClusteringOptimizer",
    
    # Registry
    "algorithm_registry",
    
    # Types
    "OptimizationResult",
    "NoiseAnalysis",
    "ConvergenceStatus",
    "AlgorithmPerformance",
    "ParamDict",
    "OptionalParamDict",
    "ArrayLike",
]