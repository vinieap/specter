"""MiniBatch K-means clustering algorithm implementation."""

from typing import Any, Type

from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator

from .base import ClusteringAlgorithm, ParamDict


class MiniBatchKMeansAlgorithm(ClusteringAlgorithm):
    """MiniBatch K-means clustering algorithm implementation.

    This class wraps scikit-learn's MiniBatchKMeans implementation and provides
    parameter sampling and validation functionality.
    """

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return "MiniBatch K-means"

    @property
    def category(self) -> str:
        """Return the category of the algorithm."""
        return "partition"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        """Return the scikit-learn estimator class."""
        return MiniBatchKMeans

    def get_default_parameters(self) -> ParamDict:
        """Return default parameters for MiniBatch K-means."""
        return {
            "n_clusters": 8,
            "init": "k-means++",
            "batch_size": 1024,
            "max_iter": 100,
            "tol": 1e-3,
            "max_no_improvement": 10,
            "reassignment_ratio": 0.01,
        }

    def sample_parameters(self, trial: Any) -> ParamDict:
        """Sample parameters for optimization.

        Args:
            trial: An Optuna trial object for parameter sampling.

        Returns:
            A dictionary of sampled parameters.
        """
        return {
            "n_clusters": trial.suggest_int("n_clusters", 2, 20),
            "init": trial.suggest_categorical("init", ["k-means++", "random"]),
            "batch_size": trial.suggest_int("batch_size", 100, 2048),
            "max_iter": trial.suggest_int("max_iter", 50, 300),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            "max_no_improvement": trial.suggest_int("max_no_improvement", 5, 20),
            "reassignment_ratio": trial.suggest_float(
                "reassignment_ratio", 0.001, 0.1, log=True
            ),
        }

    def validate_parameters(self, parameters: ParamDict) -> bool:
        """Validate the given parameters.

        Args:
            parameters: Parameters to validate.

        Returns:
            True if parameters are valid, False otherwise.
        """
        try:
            # Check required parameters
            if "n_clusters" not in parameters:
                return False

            # Validate n_clusters
            n_clusters = parameters["n_clusters"]
            if not isinstance(n_clusters, int) or n_clusters < 2:
                return False

            # Validate init
            init = parameters.get("init", "k-means++")
            if init not in ["k-means++", "random"]:
                return False

            # Validate batch_size
            batch_size = parameters.get("batch_size", 1024)
            if not isinstance(batch_size, int) or batch_size < 1:
                return False

            # Validate max_iter
            max_iter = parameters.get("max_iter", 100)
            if not isinstance(max_iter, int) or max_iter < 1:
                return False

            # Validate tol
            tol = parameters.get("tol", 1e-3)
            if not isinstance(tol, (int, float)) or tol <= 0:
                return False

            # Validate max_no_improvement
            max_no_improvement = parameters.get("max_no_improvement", 10)
            if not isinstance(max_no_improvement, int) or max_no_improvement < 0:
                return False

            # Validate reassignment_ratio
            reassignment_ratio = parameters.get("reassignment_ratio", 0.01)
            if (
                not isinstance(reassignment_ratio, (int, float))
                or not 0 <= reassignment_ratio <= 1
            ):
                return False

            return True
        except (TypeError, ValueError):
            return False
