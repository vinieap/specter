"""Parameter definitions and sampling for different clustering algorithms."""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import (
    SpectralClustering,
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    MeanShift,
    OPTICS,
)

def get_algorithm_param_sampler(algorithm: str):
    """Get the parameter sampling function for a specific algorithm."""
    samplers = {
        'spectral': sample_spectral_params,
        'kmeans': sample_kmeans_params,
        'dbscan': sample_dbscan_params,
        'agglomerative': sample_agglomerative_params,
        'birch': sample_birch_params,
        'meanshift': sample_meanshift_params,
        'optics': sample_optics_params,
    }
    return samplers.get(algorithm.lower())

def get_algorithm_class(algorithm: str):
    """Get the scikit-learn class for a specific algorithm."""
    algorithms = {
        'spectral': SpectralClustering,
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering,
        'birch': Birch,
        'meanshift': MeanShift,
        'optics': OPTICS,
    }
    return algorithms.get(algorithm.lower())

def sample_spectral_params(trial):
    """Sample parameters for Spectral Clustering."""
    return {
        "affinity": trial.suggest_categorical(
            "affinity", ["rbf", "polynomial", "nearest_neighbors"]
        ),
        "gamma": trial.suggest_float("gamma", 0.001, 10.0, log=True),
        "n_clusters": trial.suggest_int("n_clusters", 2, 18),
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
        "eigen_solver": trial.suggest_categorical(
            "eigen_solver", ["arpack", "lobpcg", "amg"]
        ),
        "assign_labels": trial.suggest_categorical(
            "assign_labels", ["kmeans", "discretize", "cluster_qr"]
        ),
        "n_components_factor": trial.suggest_int("n_components_factor", 3, 20),
        "n_init": trial.suggest_int("n_init", 5, 15),
    }

def sample_kmeans_params(trial):
    """Sample parameters for K-Means clustering."""
    return {
        "n_clusters": trial.suggest_int("n_clusters", 2, 18),
        "init": trial.suggest_categorical("init", ["k-means++", "random"]),
        "n_init": trial.suggest_int("n_init", 5, 15),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
    }

def sample_dbscan_params(trial):
    """Sample parameters for DBSCAN clustering."""
    return {
        "eps": trial.suggest_float("eps", 0.01, 2.0, log=True),
        "min_samples": trial.suggest_int("min_samples", 2, 10),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
    }

def sample_agglomerative_params(trial):
    """Sample parameters for Agglomerative clustering."""
    return {
        "n_clusters": trial.suggest_int("n_clusters", 2, 18),
        "affinity": trial.suggest_categorical(
            "affinity", ["euclidean", "manhattan", "cosine"]
        ),
        "linkage": trial.suggest_categorical(
            "linkage", ["ward", "complete", "average", "single"]
        ),
    }

def sample_birch_params(trial):
    """Sample parameters for Birch clustering."""
    return {
        "n_clusters": trial.suggest_int("n_clusters", 2, 18),
        "threshold": trial.suggest_float("threshold", 0.1, 1.0),
        "branching_factor": trial.suggest_int("branching_factor", 10, 100),
    }

def sample_meanshift_params(trial):
    """Sample parameters for Mean Shift clustering."""
    return {
        "bandwidth": trial.suggest_float("bandwidth", 0.1, 5.0, log=True),
        "bin_seeding": trial.suggest_categorical("bin_seeding", [True, False]),
        "cluster_all": trial.suggest_categorical("cluster_all", [True, False]),
    }

def sample_optics_params(trial):
    """Sample parameters for OPTICS clustering."""
    return {
        "min_samples": trial.suggest_int("min_samples", 2, 10),
        "max_eps": trial.suggest_float("max_eps", 0.1, 5.0, log=True),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
        "cluster_method": trial.suggest_categorical(
            "cluster_method", ["xi", "dbscan"]
        ),
    }

def prepare_clusterer_params(params: Dict[str, Any], X: np.ndarray, algorithm: str) -> Dict[str, Any]:
    """Prepare parameters for the clustering algorithm by handling conversions and cleanup."""
    if algorithm.lower() == 'spectral':
        return prepare_spectral_params(params, X)
    return params.copy()

def prepare_spectral_params(params: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    """Prepare parameters specifically for Spectral Clustering."""
    # Handle n_components conversion
    if params.get("n_components_factor", 8) == params["n_clusters"]:
        params["n_components"] = None
    else:
        params["n_components"] = min(params.get("n_components_factor", 8), len(X) - 1)

    # Remove the factor parameter before passing to SpectralClustering
    params_clean = {k: v for k, v in params.items() if k != "n_components_factor"}

    # Handle affinity-specific parameters
    if params["affinity"] == "nearest_neighbors":
        # Remove gamma for nearest_neighbors
        params_clean = {k: v for k, v in params_clean.items() if k != "gamma"}
    elif params["affinity"] in ["rbf", "polynomial"]:
        # Remove n_neighbors for kernel methods
        params_clean = {k: v for k, v in params_clean.items() if k != "n_neighbors"}

    return params_clean