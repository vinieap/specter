"""
Spectral Clustering Optimization Package

A high-performance Bayesian optimization framework for spectral clustering
with parallel evaluation and real-time visualization capabilities.

Main Components:
- BatchBayesianSpectralOptimizer: High-performance batch optimization
- BayesianSpectralOptimizer: Traditional sequential optimization
- optimize_spectral_clustering: Main API function
- Visualization tools for analysis and comparison

Example Usage:
    from spectral_clustering import optimize_spectral_clustering

    results = optimize_spectral_clustering(
        X,
        n_calls=100,
        use_batch_optimizer=True,
        use_dashboard=True
    )

    clusterer = results['best_clusterer']
    labels = clusterer.fit_predict(X)
"""

from .api import optimize_spectral_clustering
from .batch_optimizer import BatchBayesianSpectralOptimizer
from .config import DEFAULT_VERBOSITY, N_CORES, PARAM_NAMES, VerbosityLevel
from .evaluation import evaluate_params_worker, parallel_objective_function
from .parameters import prepare_clusterer_params, sample_params
from .sequential_optimizer import BayesianSpectralOptimizer
from .utils import format_params_for_display, get_array_size_mb, get_memory_usage
from .visualization import (
    create_optimization_summary_plots,
    generate_optimization_visualizations,
)

__version__ = "1.0.0"
__author__ = "Spectral Clustering Optimization Team"

__all__ = [
    # Main API
    "optimize_spectral_clustering",
    # Optimizers
    "BatchBayesianSpectralOptimizer",
    "BayesianSpectralOptimizer",
    # Configuration
    "VerbosityLevel",
    "DEFAULT_VERBOSITY",
    "N_CORES",
    "PARAM_NAMES",
    # Utilities
    "get_memory_usage",
    "get_array_size_mb",
    "format_params_for_display",
    # Parameters
    "sample_params",
    "prepare_clusterer_params",
    # Evaluation
    "evaluate_params_worker",
    "parallel_objective_function",
    # Visualization
    "generate_optimization_visualizations",
    "create_optimization_summary_plots",
]
