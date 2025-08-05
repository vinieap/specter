"""KMeans clustering algorithm implementation."""

from typing import Type

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from .base import ClusteringAlgorithm, ParamDict
from ..core.config import config


class KMeansAlgorithm(ClusteringAlgorithm):
    """KMeans clustering algorithm.

    This is the standard k-means algorithm that minimizes within-cluster
    variance by iteratively updating cluster centers.
    """

    @property
    def name(self) -> str:
        return "kmeans"

    @property
    def category(self) -> str:
        return "basic"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return KMeans

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for KMeans.

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
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
        }

    def sample_parameters(self, trial: optuna.Trial) -> ParamDict:
        """Sample parameters for KMeans using Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters using configuration ranges
        """
        return {
            "n_clusters": trial.suggest_int(
                "n_clusters",
                config.algorithm.kmeans.min_clusters,
                config.algorithm.kmeans.max_clusters,
            ),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "n_init": trial.suggest_int("n_init", 5, 20),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for KMeans.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required_params = {"n_clusters", "init", "n_init", "max_iter", "tol"}
        missing = required_params - set(params.keys())
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

        if params["init"] not in ["k-means++", "random"]:
            raise ValueError("init must be 'k-means++' or 'random'")

        if not isinstance(params["n_init"], int) or params["n_init"] < 1:
            raise ValueError("n_init must be an integer >= 1")

        if not isinstance(params["max_iter"], int) or params["max_iter"] < 1:
            raise ValueError("max_iter must be an integer >= 1")

        if not isinstance(params["tol"], (int, float)) or params["tol"] <= 0:
            raise ValueError("tol must be a positive number")

    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters for KMeans based on input data.

        Ensures n_clusters is not larger than the number of samples.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to prepare
        X : array-like
            Input data to cluster

        Returns
        -------
        Dict[str, Any]
            Prepared parameters
        """
        prepared = params.copy()
        n_samples = X.shape[0]

        if prepared["n_clusters"] > n_samples:
            prepared["n_clusters"] = n_samples

        return prepared
