from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from .config import N_CORES, VerbosityLevel
from .parameters import prepare_clusterer_params
from .utils import format_params_for_display


def evaluate_params_worker(params_data):
    """Worker function for evaluating parameters in parallel"""
    X, params, verbosity = params_data

    try:
        # Prepare parameters for SpectralClustering
        params_clean = prepare_clusterer_params(params, X)

        # Clean display parameters
        if verbosity >= VerbosityLevel.DETAILED:
            display_params = format_params_for_display(params_clean)
            print(f"    → Testing: {display_params}")

        clusterer = SpectralClustering(**params_clean, random_state=42)
        labels = clusterer.fit_predict(X)

        # Check if clustering was successful
        n_clusters_found = len(np.unique(labels))
        if n_clusters_found <= 1:
            return -1.0, False, "Single cluster result"

        score = silhouette_score(X, labels)
        return score, True, f"Score: {score:.4f}, Clusters: {n_clusters_found}"

    except Exception as e:
        return -1.0, False, f"Error: {str(e)[:50]}"


def parallel_objective_function(
    X, params_list, n_jobs=None, verbosity=VerbosityLevel.DETAILED
):
    """Evaluate multiple parameter combinations in parallel"""
    if n_jobs is None:
        n_jobs = N_CORES

    work_items = [(X, params, verbosity) for params in params_list]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(evaluate_params_worker, item) for item in work_items]
        results = []

        for future in as_completed(futures, timeout=300):
            try:
                score, success, message = future.result(timeout=100)
                results.append((score, success, message))
                if verbosity >= VerbosityLevel.DETAILED:
                    if success:
                        print(f"      ✓ {message}")
                    else:
                        print(f"      ✗ Failed: {message}")
            except Exception as e:
                results.append((-1.0, False, f"Worker timeout/error: {str(e)}"))

    return results
