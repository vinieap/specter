"""
Backwards Compatibility Layer

This module provides backwards compatibility for older versions of the API.
It maintains the old function signatures while internally using the new API.

Warning: These functions are deprecated and will be removed in a future version.
         Please use the new API functions from clustering.api instead.
"""

import warnings
from typing import Dict, Any, Optional

import numpy as np

from .api import (
    optimize_clustering,
    analyze_clusters,
    evaluate_clustering,
    compare_algorithms
)


def optimize_spectral_clustering(
    X,
    n_calls=100,
    n_jobs=None,
    batch_size=None,
    verbosity=1,
    use_batch_optimizer=True,
    use_dashboard=False,
    dashboard_port=8080,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Backwards compatible version of spectral clustering optimization.
    
    Warning: This function is deprecated. Use optimize_clustering() instead.
    """
    warnings.warn(
        "optimize_spectral_clustering() is deprecated and will be removed in "
        "a future version. Please use optimize_clustering(algorithm='spectral') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    results = optimize_clustering(
        X=X,
        algorithm="spectral",
        n_calls=n_calls,
        n_jobs=n_jobs,
        batch_size=batch_size,
        verbosity=verbosity,
        use_batch_optimizer=use_batch_optimizer,
        use_dashboard=use_dashboard,
        dashboard_port=dashboard_port,
        **optimizer_kwargs
    )
    
    # Convert new OptimizationResult to old dict format
    return {
        "best_score": results.best_score,
        "best_params": results.best_params,
        "best_model": results.best_model,
        "history": results.history,
        "convergence_info": results.convergence_info,
        "execution_stats": results.execution_stats
    }


def optimize_kmeans(
    X,
    n_calls=100,
    n_jobs=None,
    batch_size=None,
    verbosity=1,
    use_batch_optimizer=True,
    use_dashboard=False,
    dashboard_port=8080,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Backwards compatible version of KMeans optimization.
    
    Warning: This function is deprecated. Use optimize_clustering() instead.
    """
    warnings.warn(
        "optimize_kmeans() is deprecated and will be removed in "
        "a future version. Please use optimize_clustering(algorithm='kmeans') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    results = optimize_clustering(
        X=X,
        algorithm="kmeans",
        n_calls=n_calls,
        n_jobs=n_jobs,
        batch_size=batch_size,
        verbosity=verbosity,
        use_batch_optimizer=use_batch_optimizer,
        use_dashboard=use_dashboard,
        dashboard_port=dashboard_port,
        **optimizer_kwargs
    )
    
    return {
        "best_score": results.best_score,
        "best_params": results.best_params,
        "best_model": results.best_model,
        "history": results.history,
        "convergence_info": results.convergence_info,
        "execution_stats": results.execution_stats
    }


def analyze_noise(
    X,
    model,
    random_state=None
) -> Dict[str, Any]:
    """
    Backwards compatible version of noise analysis.
    
    Warning: This function is deprecated. Use analyze_clusters() instead.
    """
    warnings.warn(
        "analyze_noise() is deprecated and will be removed in "
        "a future version. Please use analyze_clusters() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    results = analyze_clusters(
        X=X,
        model=model,
        noise_analysis=True,
        convergence_analysis=False,
        stability_analysis=False,
        random_state=random_state
    )
    
    return results.get("noise_analysis", {})


def analyze_convergence(
    model
) -> Dict[str, Any]:
    """
    Backwards compatible version of convergence analysis.
    
    Warning: This function is deprecated. Use analyze_clusters() instead.
    """
    warnings.warn(
        "analyze_convergence() is deprecated and will be removed in "
        "a future version. Please use analyze_clusters() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    results = analyze_clusters(
        X=np.array([]),  # Not used for convergence analysis
        model=model,
        noise_analysis=False,
        convergence_analysis=True,
        stability_analysis=False
    )
    
    return results.get("convergence_info", {})


# Update __init__.py to expose old functions
__all__ = [
    'optimize_spectral_clustering',
    'optimize_kmeans',
    'analyze_noise',
    'analyze_convergence'
]