"""OPTICS clustering algorithm implementation."""

from typing import Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import OPTICS

from .base import ClusteringAlgorithm, ParamDict


class OPTICSAlgorithm(ClusteringAlgorithm):
    """OPTICS (Ordering Points To Identify the Clustering Structure).

    OPTICS is similar to DBSCAN but addresses one of DBSCAN's weaknesses:
    the inability to detect meaningful clusters in data of varying density.
    """

    @property
    def name(self) -> str:
        return "optics"

    @property
    def category(self) -> str:
        return "density"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return OPTICS

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for OPTICS.

        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - min_samples: 5
            - max_eps: np.inf
            - metric: 'euclidean'
            - cluster_method: 'xi'
            - eps: None
            - xi: 0.05
            - min_cluster_size: 5 (must be <= min_samples)
            - leaf_size: 30
        """
        return {
            "min_samples": 5,
            "max_eps": np.inf,
            "metric": "euclidean",
            "cluster_method": "xi",
            "eps": None,
            "xi": 0.05,
            "min_cluster_size": 5,  # Must be <= min_samples
            "leaf_size": 30,
        }

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for OPTICS.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - min_samples: int in [2, 10]
            - max_eps: float in [0.5, 2.0]
            - metric: one of ['euclidean', 'manhattan', 'cosine']
            - cluster_method: one of ['xi', 'dbscan']
            - xi: float in [0.01, 0.1]
            - min_cluster_size: int in [2, 10]
            - leaf_size: int in [20, 50]
        """
        params = {
            "min_samples": trial.suggest_int("min_samples", 2, 10),
            "max_eps": trial.suggest_float("max_eps", 0.5, 2.0),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
            "cluster_method": trial.suggest_categorical(
                "cluster_method", ["xi", "dbscan"]
            ),
            "eps": None,  # Let OPTICS determine this
            "leaf_size": trial.suggest_int("leaf_size", 20, 50),
        }

        if params["cluster_method"] == "xi":
            params["xi"] = trial.suggest_float("xi", 0.01, 0.1)
            params["min_cluster_size"] = trial.suggest_int("min_cluster_size", 2, 10)
        else:
            params["xi"] = 0.05  # Default value
            params["min_cluster_size"] = None

        return params

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for OPTICS.

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
            "min_samples",
            "max_eps",
            "metric",
            "cluster_method",
            "eps",
            "xi",
            "min_cluster_size",
            "leaf_size",
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if not isinstance(params["min_samples"], int) or params["min_samples"] < 2:
            raise ValueError("min_samples must be an integer >= 2")

        if not isinstance(params["max_eps"], (int, float)) or params["max_eps"] <= 0:
            raise ValueError("max_eps must be a positive number")

        if params["metric"] not in ["euclidean", "manhattan", "cosine"]:
            raise ValueError(
                "metric must be one of: 'euclidean', 'manhattan', 'cosine'"
            )

        if params["cluster_method"] not in ["xi", "dbscan"]:
            raise ValueError("cluster_method must be one of: 'xi', 'dbscan'")

        if params["cluster_method"] == "xi":
            if not isinstance(params["xi"], (int, float)) or not 0 < params["xi"] < 1:
                raise ValueError("xi must be a number between 0 and 1")

            if (
                not isinstance(params["min_cluster_size"], int)
                or params["min_cluster_size"] < 2
            ):
                raise ValueError("min_cluster_size must be an integer >= 2")

        if not isinstance(params["leaf_size"], int) or params["leaf_size"] < 1:
            raise ValueError("leaf_size must be an integer >= 1")

    def supports_predict(self) -> bool:
        """OPTICS does not support predicting on new data.

        Returns
        -------
        bool
            Always False for OPTICS
        """
        return False
