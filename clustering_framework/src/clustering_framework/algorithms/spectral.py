"""Spectral clustering algorithm implementation."""

from typing import Type

from sklearn.base import BaseEstimator
from sklearn.cluster import SpectralClustering

from .base import ClusteringAlgorithm, ParamDict
from ..core.config import config


class SpectralClusteringAlgorithm(ClusteringAlgorithm):
    """Spectral Clustering.

    This algorithm performs dimensionality reduction using the eigenvalues of a
    similarity matrix, then applies k-means in the reduced space. It's particularly
    good at finding non-spherical clusters.
    """

    @property
    def name(self) -> str:
        return "spectral"

    @property
    def category(self) -> str:
        return "affinity"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return SpectralClustering

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Spectral Clustering.

        Returns
        -------
        Dict[str, Any]
            Default parameters from configuration
        """
        return {
            "n_clusters": (
                config.algorithm.kmeans.max_clusters
                + config.algorithm.kmeans.min_clusters
            )
            // 2,  # Middle of range
            "eigen_solver": None,
            "n_components": None,
            "n_init": 10,
            "gamma": 1.0,
            "affinity": "rbf",
            "n_neighbors": (
                config.algorithm.spectral.max_n_neighbors
                + config.algorithm.spectral.min_n_neighbors
            )
            // 2,  # Middle of range
            "eigen_tol": 0.0,
            "assign_labels": "kmeans",
            "degree": 3,
            "coef0": 1,
            "kernel_params": None,
            "n_jobs": None,
        }

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Spectral Clustering.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters using configuration ranges
        """
        params = {
            "n_clusters": trial.suggest_int(
                "n_clusters",
                config.algorithm.kmeans.min_clusters,
                config.algorithm.kmeans.max_clusters,
            ),
            "n_init": trial.suggest_int("n_init", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.1, 10.0, log=True),
            "affinity": trial.suggest_categorical(
                "affinity", ["rbf", "nearest_neighbors"]
            ),
            "n_neighbors": trial.suggest_int(
                "n_neighbors",
                config.algorithm.spectral.min_n_neighbors,
                config.algorithm.spectral.max_n_neighbors,
            ),
            "assign_labels": trial.suggest_categorical(
                "assign_labels", ["kmeans", "discretize"]
            ),
        }

        # No additional parameters needed for rbf affinity

        return params

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Spectral Clustering.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required = {
            "n_clusters",
            "n_init",
            "gamma",
            "affinity",
            "assign_labels",
            "n_neighbors",
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if not isinstance(params["n_clusters"], int):
            raise ValueError("n_clusters must be an integer")
        if params["n_clusters"] < config.algorithm.kmeans.min_clusters:
            raise ValueError(
                f"n_clusters must be >= {config.algorithm.kmeans.min_clusters}"
            )
        if params["n_clusters"] > config.algorithm.kmeans.max_clusters:
            raise ValueError(
                f"n_clusters must be <= {config.algorithm.kmeans.max_clusters}"
            )

        if not isinstance(params["n_init"], int) or params["n_init"] < 1:
            raise ValueError("n_init must be an integer >= 1")

        if not isinstance(params["gamma"], (int, float)) or params["gamma"] <= 0:
            raise ValueError("gamma must be a positive number")

        if params["affinity"] not in ["rbf", "nearest_neighbors"]:
            raise ValueError("affinity must be one of: 'rbf', 'nearest_neighbors'")

        if not isinstance(params["n_neighbors"], int):
            raise ValueError("n_neighbors must be an integer")
        if params["n_neighbors"] < config.algorithm.spectral.min_n_neighbors:
            raise ValueError(
                f"n_neighbors must be >= {config.algorithm.spectral.min_n_neighbors}"
            )
        if params["n_neighbors"] > config.algorithm.spectral.max_n_neighbors:
            raise ValueError(
                f"n_neighbors must be <= {config.algorithm.spectral.max_n_neighbors}"
            )

        if params["assign_labels"] not in ["kmeans", "discretize"]:
            raise ValueError("assign_labels must be one of: 'kmeans', 'discretize'")
