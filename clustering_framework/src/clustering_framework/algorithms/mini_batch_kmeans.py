"""Mini-batch KMeans implementation."""

from typing import Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans

from .base import ClusteringAlgorithm, ParamDict


class MiniBatchKMeansAlgorithm(ClusteringAlgorithm):
    """Mini-batch KMeans clustering algorithm.

    This variant uses mini-batches to reduce computation time, which is useful
    for large datasets. It trades off accuracy for speed.
    """

    @property
    def name(self) -> str:
        return "mini_batch_kmeans"

    @property
    def category(self) -> str:
        return "basic"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return MiniBatchKMeans

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Mini-batch KMeans.

        Returns
        -------
        Dict[str, Any]
            Default parameters including mini-batch specific ones:
            - batch_size: 1024
            - max_no_improvement: 10
        """
        params = {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-4,
            "batch_size": 1024,
            "max_no_improvement": 10,
        }
        return params

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Mini-batch KMeans.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters including mini-batch specific ones:
            - batch_size: int in [256, 4096]
            - max_no_improvement: int in [5, 20]
        """
        params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "n_init": trial.suggest_int("n_init", 5, 20),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_int("batch_size", 256, 4096),
            "max_no_improvement": trial.suggest_int("max_no_improvement", 5, 20),
        }
        return params

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Mini-batch KMeans.

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
            "batch_size",
            "max_no_improvement",
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

        if not isinstance(params["batch_size"], int) or params["batch_size"] < 1:
            raise ValueError("batch_size must be an integer >= 1")

        if (
            not isinstance(params["max_no_improvement"], int)
            or params["max_no_improvement"] < 0
        ):
            raise ValueError("max_no_improvement must be an integer >= 0")

    def prepare_parameters(self, params: ParamDict, X: np.ndarray) -> ParamDict:
        """Prepare parameters for Mini-batch KMeans.

        Ensures batch_size is not larger than the dataset size.

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

        if prepared["batch_size"] > n_samples:
            prepared["batch_size"] = n_samples

        return prepared
