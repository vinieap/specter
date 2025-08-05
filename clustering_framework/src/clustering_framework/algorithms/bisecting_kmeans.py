"""Bisecting KMeans implementation."""

from typing import Type

from sklearn.base import BaseEstimator
from sklearn.cluster import BisectingKMeans

from .base import ClusteringAlgorithm, ParamDict


class BisectingKMeansAlgorithm(ClusteringAlgorithm):
    """Bisecting KMeans clustering algorithm.

    This variant uses a divisive hierarchical strategy, recursively splitting
    clusters using standard k-means with k=2. This can produce more balanced
    clusters than standard k-means.
    """

    @property
    def name(self) -> str:
        return "bisecting_kmeans"

    @property
    def category(self) -> str:
        return "basic"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return BisectingKMeans

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Bisecting KMeans.

        Returns
        -------
        Dict[str, Any]
            Default parameters including bisecting specific ones:
            - bisecting_strategy: 'biggest_inertia'
        """
        params = {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "bisecting_strategy": "biggest_inertia",
        }
        return params

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Bisecting KMeans.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters including bisecting specific ones:
            - bisecting_strategy: one of ['biggest_inertia', 'largest_cluster']
        """
        params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "n_init": trial.suggest_int("n_init", 5, 20),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
            "bisecting_strategy": trial.suggest_categorical(
                "bisecting_strategy", ["biggest_inertia", "largest_cluster"]
            ),
        }
        return params

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Bisecting KMeans.

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
            "init",
            "n_init",
            "max_iter",
            "tol",
            "bisecting_strategy",
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if not isinstance(params["n_clusters"], int) or params["n_clusters"] < 2:
            raise ValueError("n_clusters must be an integer >= 2")

        if params["init"] not in ["k-means++", "random"]:
            raise ValueError("init must be 'k-means++' or 'random'")

        if not isinstance(params["n_init"], int) or params["n_init"] < 1:
            raise ValueError("n_init must be an integer >= 1")

        if not isinstance(params["max_iter"], int) or params["max_iter"] < 1:
            raise ValueError("max_iter must be an integer >= 1")

        if not isinstance(params["tol"], (int, float)) or params["tol"] <= 0:
            raise ValueError("tol must be a positive number")

        if params["bisecting_strategy"] not in ["biggest_inertia", "largest_cluster"]:
            raise ValueError(
                "bisecting_strategy must be one of: 'biggest_inertia', 'largest_cluster'"
            )
