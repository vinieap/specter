"""BIRCH clustering algorithm implementation."""

from typing import Type

from sklearn.base import BaseEstimator
from sklearn.cluster import Birch

from .base import ClusteringAlgorithm, ParamDict


class BirchAlgorithm(ClusteringAlgorithm):
    """BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

    BIRCH builds a tree structure called CFTree to incrementally cluster the data.
    It's particularly effective for large datasets as it only needs one pass over
    the data to create a good clustering.
    """

    @property
    def name(self) -> str:
        return "birch"

    @property
    def category(self) -> str:
        return "hierarchical"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return Birch

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for BIRCH.

        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - threshold: 0.3 (lower value to allow for sufficient subclusters)
            - branching_factor: 50
            - n_clusters: 3
            - compute_labels: True
        """
        n_clusters = 3
        return {
            "threshold": 0.3,  # Set below 1/n_clusters (0.33) to ensure enough subclusters
            "branching_factor": 50,
            "n_clusters": n_clusters,
            "compute_labels": True,
        }

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for BIRCH.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - threshold: float in [0.01, 0.5]  # Lower range to avoid subcluster warnings
            - branching_factor: int in [10, 100]
            - n_clusters: int in [2, 20]
            - compute_labels: bool
        """
        n_clusters = trial.suggest_int("n_clusters", 2, 20)
        # Ensure threshold is small enough to allow for sufficient subclusters
        max_threshold = min(0.5, 1.0 / n_clusters)

        return {
            "threshold": trial.suggest_float("threshold", 0.01, max_threshold),
            "branching_factor": trial.suggest_int("branching_factor", 10, 100),
            "n_clusters": n_clusters,
            "compute_labels": trial.suggest_categorical(
                "compute_labels", [True, False]
            ),
        }

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for BIRCH.

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
            "threshold",
            "branching_factor",
            "n_clusters",
            "compute_labels",
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if (
            not isinstance(params["threshold"], (int, float))
            or params["threshold"] <= 0
            or params["threshold"] > 1.0
        ):
            raise ValueError("threshold must be a number between 0 and 1")

        if (
            not isinstance(params["branching_factor"], int)
            or params["branching_factor"] < 2
        ):
            raise ValueError("branching_factor must be an integer >= 2")

        if not isinstance(params["n_clusters"], int) or params["n_clusters"] < 2:
            raise ValueError("n_clusters must be an integer >= 2")

        if not isinstance(params["compute_labels"], bool):
            raise ValueError("compute_labels must be a boolean")

        # Validate threshold is appropriate for n_clusters
        if params["threshold"] > 1.0 / params["n_clusters"]:
            raise ValueError(
                f"threshold {params['threshold']} is too high for {params['n_clusters']} clusters. "
                f"Should be <= {1.0 / params['n_clusters']}"
            )
