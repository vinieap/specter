def sample_params(trial):
    """Sample parameters using Optuna's trial object"""
    params = {
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
    return params


def prepare_clusterer_params(params, X):
    """Prepare parameters for SpectralClustering by handling conversions and cleanup"""
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
