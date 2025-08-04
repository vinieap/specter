"""Evaluation functions for clustering optimization."""
import multiprocessing
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings

from .algorithms.base import BaseClusteringAlgorithm
from .config import VerbosityLevel


def evaluate_clustering(
    X: np.ndarray,
    params: Dict[str, Any],
    algorithm: BaseClusteringAlgorithm,
) -> Tuple[float, bool, str]:
    """
    Evaluate a clustering configuration and return the score.

    Parameters
    ----------
    X : array-like
        Input data to cluster
    params : dict
        Parameters for the clustering algorithm
    algorithm : BaseClusteringAlgorithm
        The clustering algorithm to use

    Returns
    -------
    score : float
        Silhouette score (-1 if clustering failed)
    success : bool
        Whether clustering completed successfully
    message : str
        Error message if clustering failed, empty string otherwise
    """
    try:
        # Prepare parameters for the specific algorithm
        params_clean = algorithm.prepare_parameters(params, X)
        
        # Create and fit the clusterer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            clusterer = algorithm.create_estimator(params_clean)
            labels = clusterer.fit_predict(X)

        # Handle algorithms that may return -1 for noise points (like DBSCAN)
        unique_labels = np.unique(labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        
        if n_clusters < 2:
            return -1.0, False, "Less than 2 clusters found"
            
        # For algorithms with noise points, we only compute silhouette score
        # on the points that were assigned to clusters
        if -1 in unique_labels:
            mask = labels != -1
            if np.sum(mask) < 2:
                return -1.0, False, "Not enough points assigned to clusters"
            score = silhouette_score(X[mask], labels[mask])
        else:
            score = silhouette_score(X, labels)

        return float(score), True, ""

    except Exception as e:
        return -1.0, False, str(e)


def parallel_objective_function(
    X: np.ndarray,
    params_list: List[Dict[str, Any]],
    algorithm: BaseClusteringAlgorithm,
    n_jobs: int,
    verbosity: int = VerbosityLevel.MINIMAL,
) -> List[Tuple[float, bool, str]]:
    """
    Evaluate multiple parameter sets in parallel.

    Parameters
    ----------
    X : array-like
        Input data to cluster
    params_list : list of dict
        List of parameter sets to evaluate
    algorithm : BaseClusteringAlgorithm
        The clustering algorithm to use
    n_jobs : int
        Number of parallel jobs
    verbosity : int
        Verbosity level

    Returns
    -------
    list of tuples
        Each tuple contains (score, success, message) for each parameter set
    """
    # Create a pool of workers
    with multiprocessing.Pool(n_jobs) as pool:
        # Prepare arguments for parallel execution
        args = [(X, params, algorithm) for params in params_list]
        
        # Execute evaluations in parallel
        results = pool.starmap(evaluate_clustering, args)

    return results