"""
Parameter Validation

This module provides validation functions for API parameters.
It ensures that all inputs meet the required specifications and
provides helpful error messages when validation fails.
"""

from typing import Any, Dict, List, Optional, Union
import numbers

import numpy as np
from sklearn.base import BaseEstimator


def validate_array(
    X: Any,
    allow_nd: bool = False,
    min_samples: int = 1,
    min_features: int = 1
) -> np.ndarray:
    """
    Validate input array X.

    Parameters
    ----------
    X : array-like
        Input data to validate
    allow_nd : bool, default=False
        Whether to allow n-dimensional arrays (n>2)
    min_samples : int, default=1
        Minimum number of samples required
    min_features : int, default=1
        Minimum number of features required

    Returns
    -------
    np.ndarray
        Validated array

    Raises
    ------
    ValueError
        If validation fails
    """
    # Convert to numpy array if needed
    if not isinstance(X, np.ndarray):
        try:
            X = np.asarray(X)
        except Exception as e:
            raise ValueError(f"Could not convert X to numpy array: {str(e)}")

    # Check dimensions
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2 and not allow_nd:
        raise ValueError(
            f"Expected 2D array, got {X.ndim}D array instead. "
            "Reshape your data using array.reshape(-1, 1) if "
            "it contains a single feature"
        )

    # Check size
    n_samples, n_features = X.shape[0], X.shape[1]
    if n_samples < min_samples:
        raise ValueError(
            f"Found array with {n_samples} sample(s) "
            f"but a minimum of {min_samples} is required"
        )
    if n_features < min_features:
        raise ValueError(
            f"Found array with {n_features} feature(s) "
            f"but a minimum of {min_features} is required"
        )

    return X


def validate_estimator(estimator: Any) -> BaseEstimator:
    """
    Validate clustering estimator.

    Parameters
    ----------
    estimator : Any
        Estimator to validate

    Returns
    -------
    BaseEstimator
        Validated estimator

    Raises
    ------
    ValueError
        If validation fails
    """
    if not isinstance(estimator, BaseEstimator):
        raise ValueError(
            "Estimator must be a scikit-learn BaseEstimator instance"
        )
    
    required_methods = ['fit', 'predict']
    missing_methods = [
        method for method in required_methods
        if not hasattr(estimator, method)
    ]
    
    if missing_methods:
        raise ValueError(
            f"Estimator missing required methods: {missing_methods}"
        )
    
    return estimator


def validate_algorithm(algorithm: str, available_algorithms: List[str]) -> str:
    """
    Validate clustering algorithm name.

    Parameters
    ----------
    algorithm : str
        Algorithm name to validate
    available_algorithms : List[str]
        List of available algorithm names

    Returns
    -------
    str
        Validated algorithm name

    Raises
    ------
    ValueError
        If validation fails
    """
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm name must be a string")
    
    algorithm = algorithm.lower()
    if algorithm not in available_algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available algorithms: {available_algorithms}"
        )
    
    return algorithm


def validate_optimizer_params(
    n_calls: int,
    n_jobs: Optional[int],
    batch_size: Optional[int],
    verbosity: int,
    use_batch_optimizer: bool,
    use_dashboard: bool,
    dashboard_port: int
) -> Dict[str, Any]:
    """
    Validate optimizer parameters.

    Parameters
    ----------
    n_calls : int
        Number of optimization calls
    n_jobs : int, optional
        Number of parallel jobs
    batch_size : int, optional
        Batch size for parallel evaluation
    verbosity : int
        Verbosity level
    use_batch_optimizer : bool
        Whether to use batch optimization
    use_dashboard : bool
        Whether to use dashboard
    dashboard_port : int
        Dashboard port number

    Returns
    -------
    Dict[str, Any]
        Validated parameters

    Raises
    ------
    ValueError
        If validation fails
    """
    params = {}

    # Validate n_calls
    if not isinstance(n_calls, numbers.Integral) or n_calls < 1:
        raise ValueError(f"n_calls must be a positive integer, got {n_calls}")
    params["n_calls"] = n_calls

    # Validate n_jobs
    if n_jobs is not None:
        if not isinstance(n_jobs, numbers.Integral):
            raise ValueError(f"n_jobs must be an integer, got {n_jobs}")
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be positive, got {n_jobs}")
    params["n_jobs"] = n_jobs

    # Validate batch_size
    if batch_size is not None:
        if not isinstance(batch_size, numbers.Integral):
            raise ValueError(f"batch_size must be an integer, got {batch_size}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
    params["batch_size"] = batch_size

    # Validate verbosity
    if not isinstance(verbosity, numbers.Integral):
        raise ValueError(f"verbosity must be an integer, got {verbosity}")
    params["verbosity"] = verbosity

    # Validate boolean parameters
    if not isinstance(use_batch_optimizer, bool):
        raise ValueError(
            f"use_batch_optimizer must be a boolean, got {use_batch_optimizer}"
        )
    params["use_batch_optimizer"] = use_batch_optimizer

    if not isinstance(use_dashboard, bool):
        raise ValueError(f"use_dashboard must be a boolean, got {use_dashboard}")
    params["use_dashboard"] = use_dashboard

    # Validate dashboard_port
    if not isinstance(dashboard_port, numbers.Integral):
        raise ValueError(
            f"dashboard_port must be an integer, got {dashboard_port}"
        )
    if not 1024 <= dashboard_port <= 65535:
        raise ValueError(
            f"dashboard_port must be between 1024 and 65535, got {dashboard_port}"
        )
    params["dashboard_port"] = dashboard_port

    return params


def validate_analysis_params(
    noise_analysis: bool,
    convergence_analysis: bool,
    stability_analysis: bool,
    n_stability_runs: int
) -> Dict[str, Any]:
    """
    Validate analysis parameters.

    Parameters
    ----------
    noise_analysis : bool
        Whether to perform noise analysis
    convergence_analysis : bool
        Whether to perform convergence analysis
    stability_analysis : bool
        Whether to perform stability analysis
    n_stability_runs : int
        Number of stability analysis runs

    Returns
    -------
    Dict[str, Any]
        Validated parameters

    Raises
    ------
    ValueError
        If validation fails
    """
    params = {}

    # Validate boolean parameters
    for name, value in [
        ("noise_analysis", noise_analysis),
        ("convergence_analysis", convergence_analysis),
        ("stability_analysis", stability_analysis)
    ]:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean, got {value}")
        params[name] = value

    # Validate n_stability_runs
    if not isinstance(n_stability_runs, numbers.Integral):
        raise ValueError(
            f"n_stability_runs must be an integer, got {n_stability_runs}"
        )
    if n_stability_runs < 2:
        raise ValueError(
            f"n_stability_runs must be at least 2, got {n_stability_runs}"
        )
    params["n_stability_runs"] = n_stability_runs

    return params


def validate_metrics(
    metrics: Optional[List[str]],
    available_metrics: List[str]
) -> Optional[List[str]]:
    """
    Validate metric names.

    Parameters
    ----------
    metrics : List[str], optional
        List of metric names to validate
    available_metrics : List[str]
        List of available metric names

    Returns
    -------
    List[str], optional
        Validated metric names

    Raises
    ------
    ValueError
        If validation fails
    """
    if metrics is None:
        return None

    if not isinstance(metrics, (list, tuple)):
        raise ValueError("metrics must be a list or tuple")

    invalid_metrics = set(metrics) - set(available_metrics)
    if invalid_metrics:
        raise ValueError(
            f"Unknown metrics: {invalid_metrics}. "
            f"Available metrics: {available_metrics}"
        )

    return list(metrics)