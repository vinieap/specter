"""DBSCAN clustering algorithm implementation."""

from typing import Type

import optuna
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN

from .base import ClusteringAlgorithm, ParamDict
from ..core.config import config


class DBSCANAlgorithm(ClusteringAlgorithm):
    """DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    DBSCAN finds core samples of high density and expands clusters from them.
    It's good at finding clusters of arbitrary shape and identifying noise points.
    """

    @property
    def name(self) -> str:
        return "dbscan"

    @property
    def category(self) -> str:
        return "density"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return DBSCAN

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for DBSCAN.

        Returns
        -------
        Dict[str, Any]
            Default parameters from configuration
        """
        return {
            "eps": (config.algorithm.dbscan.max_eps + config.algorithm.dbscan.min_eps)
            / 2,  # Middle of range
            "min_samples": (
                config.algorithm.dbscan.max_min_samples
                + config.algorithm.dbscan.min_min_samples
            )
            // 2,  # Middle of range
            "metric": "euclidean",
            "algorithm": "auto",
            "leaf_size": 30,
        }

    def sample_parameters(self, trial: optuna.Trial) -> ParamDict:
        """Sample parameters for DBSCAN.

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
            "eps": trial.suggest_float(
                "eps", config.algorithm.dbscan.min_eps, config.algorithm.dbscan.max_eps
            ),
            "min_samples": trial.suggest_int(
                "min_samples",
                config.algorithm.dbscan.min_min_samples,
                config.algorithm.dbscan.max_min_samples,
            ),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "cosine"]
            ),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 20, 50),
        }

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for DBSCAN.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        required = {"eps", "min_samples", "metric", "algorithm", "leaf_size"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if not isinstance(params["eps"], (int, float)):
            raise ValueError("eps must be a number")
        if params["eps"] < config.algorithm.dbscan.min_eps:
            raise ValueError(f"eps must be >= {config.algorithm.dbscan.min_eps}")
        if params["eps"] > config.algorithm.dbscan.max_eps:
            raise ValueError(f"eps must be <= {config.algorithm.dbscan.max_eps}")

        if not isinstance(params["min_samples"], int):
            raise ValueError("min_samples must be an integer")
        if params["min_samples"] < config.algorithm.dbscan.min_min_samples:
            raise ValueError(
                f"min_samples must be >= {config.algorithm.dbscan.min_min_samples}"
            )
        if params["min_samples"] > config.algorithm.dbscan.max_min_samples:
            raise ValueError(
                f"min_samples must be <= {config.algorithm.dbscan.max_min_samples}"
            )

        if params["metric"] not in ["euclidean", "manhattan", "cosine"]:
            raise ValueError(
                "metric must be one of: 'euclidean', 'manhattan', 'cosine'"
            )

        if params["algorithm"] not in ["auto", "ball_tree", "kd_tree", "brute"]:
            raise ValueError(
                "algorithm must be one of: 'auto', 'ball_tree', 'kd_tree', 'brute'"
            )

        if not isinstance(params["leaf_size"], int) or params["leaf_size"] < 1:
            raise ValueError("leaf_size must be an integer >= 1")

    def supports_predict(self) -> bool:
        """DBSCAN does not support predicting on new data.

        Returns
        -------
        bool
            Always False for DBSCAN
        """
        return False
