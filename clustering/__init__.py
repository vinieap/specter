"""
Clustering Library

A comprehensive library for clustering optimization and analysis.
"""

from .api import (
    optimize_clustering,
    analyze_clusters,
    evaluate_clustering,
    compare_algorithms
)

# Import convenience functions
from .convenience import (
    quick_cluster,
    analyze_and_improve,
    find_optimal_clusters,
    compare_and_recommend
)

# Import backwards compatibility functions
from .compat import (
    optimize_spectral_clustering,
    optimize_kmeans,
    analyze_noise,
    analyze_convergence
)

# Import core types for type hints
from .core.types import (
    OptimizationResult,
    NoiseAnalysis,
    AlgorithmPerformance
)

__version__ = "2.0.0"

__all__ = [
    # New API
    'optimize_clustering',
    'analyze_clusters',
    'evaluate_clustering',
    'compare_algorithms',
    
    # Convenience functions
    'quick_cluster',
    'analyze_and_improve',
    'find_optimal_clusters',
    'compare_and_recommend',
    
    # Core types
    'OptimizationResult',
    'NoiseAnalysis',
    'AlgorithmPerformance',
    
    # Backwards compatibility
    'optimize_spectral_clustering',
    'optimize_kmeans',
    'analyze_noise',
    'analyze_convergence'
]