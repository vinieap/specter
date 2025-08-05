"""Mean Shift clustering algorithm implementation."""

from typing import Type

from sklearn.base import BaseEstimator
from sklearn.cluster import MeanShift

from .base import ClusteringAlgorithm, ParamDict


class MeanShiftAlgorithm(ClusteringAlgorithm):
    """Mean Shift Clustering.

    This algorithm seeks modes or local maxima of density of points in feature space
    by iteratively shifting points towards the average of points in their neighborhood.
    """

    @property
    def name(self) -> str:
        return "mean_shift"

    @property
    def category(self) -> str:
        return "affinity"

    @property
    def estimator_class(self) -> Type[BaseEstimator]:
        return MeanShift

    def get_default_parameters(self) -> ParamDict:
        """Get default parameters for Mean Shift.

        Returns
        -------
        Dict[str, Any]
            Default parameters:
            - bandwidth: None (auto)
            - seeds: None
            - bin_seeding: False
            - min_bin_freq: 1
            - cluster_all: True
            - n_jobs: None
            - max_iter: 300
        """
        return {
            "bandwidth": None,
            "seeds": None,
            "bin_seeding": False,
            "min_bin_freq": 1,
            "cluster_all": True,
            "n_jobs": None,
            "max_iter": 300,
        }

    def sample_parameters(self, trial) -> ParamDict:
        """Sample parameters for Mean Shift.

        Parameters
        ----------
        trial : optuna.Trial
            Trial object for parameter sampling

        Returns
        -------
        Dict[str, Any]
            Sampled parameters:
            - bandwidth: float in [0.1, 2.0]
            - bin_seeding: bool
            - min_bin_freq: int in [1, 5]
            - cluster_all: bool
            - max_iter: int in [100, 500]
        """
        return {
            "bandwidth": trial.suggest_float("bandwidth", 0.1, 2.0),
            "bin_seeding": trial.suggest_categorical("bin_seeding", [True, False]),
            "min_bin_freq": trial.suggest_int("min_bin_freq", 1, 5),
            "cluster_all": trial.suggest_categorical("cluster_all", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "seeds": None,
            "n_jobs": None,
        }

    def validate_parameters(self, params: ParamDict) -> None:
        """Validate parameters for Mean Shift.

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
            "bandwidth",
            "seeds",
            "bin_seeding",
            "min_bin_freq",
            "cluster_all",
            "n_jobs",
            "max_iter",
        }
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        if params["bandwidth"] is not None:
            if (
                not isinstance(params["bandwidth"], (int, float))
                or params["bandwidth"] <= 0
            ):
                raise ValueError("bandwidth must be a positive number")

        if not isinstance(params["bin_seeding"], bool):
            raise ValueError("bin_seeding must be a boolean")

        if not isinstance(params["min_bin_freq"], int) or params["min_bin_freq"] < 1:
            raise ValueError("min_bin_freq must be an integer >= 1")

        if not isinstance(params["cluster_all"], bool):
            raise ValueError("cluster_all must be a boolean")

        if not isinstance(params["max_iter"], int) or params["max_iter"] < 1:
            raise ValueError("max_iter must be an integer >= 1")
