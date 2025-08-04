"""Analysis tools for clustering algorithms.

This package provides tools for analyzing clustering results, including:
- Noise detection and classification
- Convergence analysis
- Performance evaluation
"""

from .noise import NoiseAnalysis, NoiseAnalyzer
from .convergence import ConvergenceAnalysis, ConvergenceAnalyzer
from .evaluation import (
    ClusteringMetrics,
    PerformanceMetrics,
    ClusteringEvaluation,
    ClusteringEvaluator
)

__all__ = [
    'NoiseAnalysis',
    'NoiseAnalyzer',
    'ConvergenceAnalysis',
    'ConvergenceAnalyzer',
    'ClusteringMetrics',
    'PerformanceMetrics',
    'ClusteringEvaluation',
    'ClusteringEvaluator'
]