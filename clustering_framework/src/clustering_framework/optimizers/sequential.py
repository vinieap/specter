"""Sequential optimization strategy for clustering algorithms using Optuna."""

from typing import Any, Dict, Optional
import optuna
from sklearn.base import BaseEstimator

from .base import BaseOptimizer


class SequentialOptimizer(BaseOptimizer):
    """Optimizer that evaluates parameter sets sequentially using Optuna's TPE sampler."""

    def __init__(
        self,
        algorithm: Any,  # ClusteringAlgorithm
        max_trials: Optional[int] = None,
        n_startup_trials: Optional[int] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        min_trials: int = 20,
        use_dashboard: bool = False,
        dashboard_port: int = 8080,
        random_state: int = 42,
    ):
        """Initialize sequential optimizer.

        Args:
            algorithm: Clustering algorithm to optimize
            max_trials: Maximum number of optimization trials
            n_startup_trials: Number of random trials before optimization
            patience: Number of trials without improvement before convergence
            min_delta: Minimum change in score to be considered an improvement
            min_trials: Minimum number of trials before allowing convergence
            use_dashboard: Whether to start Optuna dashboard
            dashboard_port: Port for Optuna dashboard
            random_state: Random seed for reproducibility
        """
        super().__init__(
            max_trials=max_trials,
            n_startup_trials=n_startup_trials,
            patience=patience,
            min_delta=min_delta,
            min_trials=min_trials,
            use_dashboard=use_dashboard,
            dashboard_port=dashboard_port,
            random_state=random_state,
        )
        self.algorithm = algorithm

    def _create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create a model with the given parameters.

        Args:
            params: Parameters to create model with

        Returns:
            Created model instance
        """
        estimator_params = params.copy()
        if self.algorithm.supports_random_state:
            estimator_params["random_state"] = self.random_state
        return self.algorithm.estimator_class(**estimator_params)

    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters using Optuna's trial object.

        This method adapts the algorithm's parameter space to Optuna's parameter
        suggestion interface. It handles different parameter types appropriately:
        - Categorical parameters use suggest_categorical
        - Integer parameters use suggest_int
        - Float parameters use suggest_float

        Args:
            trial: Optuna trial object for parameter suggestion

        Returns:
            Dictionary of sampled parameters
        """
        return self.algorithm.sample_parameters(trial)
