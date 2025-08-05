"""Affinity Propagation clustering algorithm implementation."""

from typing import Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import AffinityPropagation

from .base import ClusteringAlgorithm, ParamDict


class AffinityPropagationAlgorithm(ClusteringAlgorithm):
    """Affinity Propagation Clustering.

    This algorithm creates clusters by passing messages between pairs of samples
    until convergence. Each message represents the suitability of one sample
    to serve as the exemplar for another sample.
    """

    @property
    def name(self) -> str:
        return "affinity_propagation"

    @property
    def category(self) -> str:
        return "affinity"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return AffinityPropagation

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Affinity Propagation.

        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - damping: 0.8 (higher value for better convergence)
            - max_iter: 500 (increased for better convergence)
            - convergence_iter: 30 (increased to ensure stable convergence)
            - preference: -5.0 (negative value for moderate number of clusters)
            - affinity: 'euclidean'
            - verbose: False
        """
        return {
            "damping": 0.8,
            "max_iter": 500,
            "convergence_iter": 30,
            "preference": -5.0,
            "affinity": "euclidean",
            "verbose": False,
        }

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Affinity Propagation.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - damping: float in [0.5, 0.95] - Higher upper bound for better convergence
            - max_iter: int in [200, 1000] - Increased iterations for convergence
            - convergence_iter: int in [15, 50] - More iterations to confirm convergence
            - affinity: one of ['euclidean', 'precomputed']
            - preference: float - Controls number of exemplars (clusters)
        """
        # Sample damping with higher values for better convergence
        damping = trial.suggest_float("damping", 0.5, 0.95)

        # Adjust max_iter based on damping - higher damping needs more iterations
        min_iter = int(
            200 * (1 + (damping - 0.5) * 2)
        )  # More iterations for higher damping
        max_iter = trial.suggest_int("max_iter", min_iter, 1000)

        # Adjust convergence_iter based on max_iter
        convergence_iter = trial.suggest_int(
            "convergence_iter", 15, min(50, max_iter // 4)
        )

        # Sample preference to control number of clusters
        # Lower values lead to fewer clusters
        preference = trial.suggest_float("preference", -10.0, 0.0)

        return {
            "damping": damping,
            "max_iter": max_iter,
            "convergence_iter": convergence_iter,
            "affinity": trial.suggest_categorical(
                "affinity", ["euclidean", "precomputed"]
            ),
            "preference": preference,
            "verbose": False,
        }

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Affinity Propagation.

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
            "damping",
            "max_iter",
            "convergence_iter",
            "preference",
            "affinity",
            "verbose",
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if (
            not isinstance(params["damping"], (int, float))
            or not 0.5 <= params["damping"] <= 0.95
        ):
            raise ValueError("damping must be a number between 0.5 and 0.95")

        if not isinstance(params["max_iter"], int) or params["max_iter"] < 200:
            raise ValueError("max_iter must be an integer >= 200")

        if (
            not isinstance(params["convergence_iter"], int)
            or params["convergence_iter"] < 15
            or params["convergence_iter"] > params["max_iter"] // 4
        ):
            raise ValueError(
                f"convergence_iter must be between 15 and {params['max_iter'] // 4}"
            )

        if params["affinity"] not in ["euclidean", "precomputed"]:
            raise ValueError("affinity must be one of: 'euclidean', 'precomputed'")

        if not isinstance(params["verbose"], bool):
            raise ValueError("verbose must be a boolean")

        if not isinstance(params["preference"], (int, float, np.ndarray)):
            raise ValueError("preference must be a number or array-like")

        # Validate that max_iter is sufficient for the given damping
        min_iter = int(200 * (1 + (params["damping"] - 0.5) * 2))
        if params["max_iter"] < min_iter:
            raise ValueError(
                f"max_iter must be >= {min_iter} for damping={params['damping']}"
            )
