"""Base optimizer interface and common optimization logic using Optuna."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional
import subprocess

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator

from ..core.config import config


@dataclass
class OptimizationResult:
    """Structured optimization results.

    Attributes:
        best_score: The best score achieved during optimization
        best_params: Parameters that achieved the best score
        best_model: The best performing model
        study: Optuna study containing full optimization history
        convergence_info: Information about optimization convergence
        execution_stats: Statistics about the optimization process
        history: List of dictionaries containing trial information
    """

    best_score: float
    best_params: Dict[str, Any]
    best_model: Any  # BaseEstimator from sklearn
    study: optuna.Study
    convergence_info: Dict[str, Any]
    execution_stats: Dict[str, Any]
    history: List[Dict[str, Any]]


class BaseOptimizer(ABC):
    """Base class for Optuna-based optimization strategies."""

    def __init__(
        self,
        max_trials: Optional[int] = None,
        n_startup_trials: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        min_trials: Optional[int] = None,
        use_dashboard: bool = False,
        dashboard_port: Optional[int] = None,
        random_state: int = 42,
    ):
        """Initialize base optimizer.

        Args:
            max_trials: Maximum number of optimization trials
            n_startup_trials: Number of random trials before optimization
            patience: Number of trials without improvement before convergence
            min_delta: Minimum change in score to be considered an improvement
            min_trials: Minimum number of trials before allowing convergence
            use_dashboard: Whether to start Optuna dashboard
            dashboard_port: Port for Optuna dashboard
            random_state: Random seed for reproducibility
        """
        # Use configuration defaults if not specified
        self.max_trials: Optional[int] = max_trials
        self.n_startup_trials: int = n_startup_trials or min(
            config.optimizer.default_n_startup_trials,
            max_trials // 4
            if max_trials
            else config.optimizer.default_n_startup_trials,
        )
        self.patience: int = patience or config.optimizer.default_patience
        self.min_delta: float = min_delta or config.optimizer.default_min_delta
        self.min_trials: int = min_trials or config.optimizer.default_min_trials
        self.use_dashboard: bool = use_dashboard
        self.dashboard_port: int = (
            dashboard_port or config.optimizer.default_dashboard_port
        )
        self.random_state: int = random_state

        self.study: Optional[optuna.Study] = None
        self.X: Optional[np.ndarray] = None
        self.best_model: Optional[BaseEstimator] = None
        self.dashboard_process: Optional[subprocess.Popen] = None

    def _setup_study(self):
        """Set up Optuna study with proper storage and sampler."""
        if self.use_dashboard:
            # Use SQLite storage for dashboard access
            storage_name = "sqlite:///optuna_clustering_studies.db"
            storage = optuna.storages.RDBStorage(url=storage_name)

            # Create unique study name with timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"clustering_study_{timestamp}_seed{self.random_state}"

            self.study = optuna.create_study(
                study_name=study_name,
                direction="maximize",  # We want to maximize clustering score
                sampler=TPESampler(
                    n_startup_trials=self.n_startup_trials, seed=self.random_state
                ),
                storage=storage,
                load_if_exists=True,
            )
        else:
            # Create in-memory study
            self.study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(
                    n_startup_trials=self.n_startup_trials, seed=self.random_state
                ),
            )

    def _start_dashboard(self):
        """Start Optuna dashboard if requested."""
        if not self.use_dashboard:
            return

        try:
            storage_url = "sqlite:///optuna_clustering_studies.db"

            # Wait for database initialization
            time.sleep(0.1)

            self.dashboard_process = subprocess.Popen(
                ["optuna-dashboard", storage_url, "--port", str(self.dashboard_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Give dashboard time to start
            time.sleep(1.0)

            print(
                f"🌐 Optuna Dashboard started at http://localhost:{self.dashboard_port}"
            )
            print("   You can monitor optimization progress in real-time!")

        except (ImportError, FileNotFoundError) as e:
            print(f"⚠️  Could not start optuna-dashboard: {e}")
            print("   Install with: pip install optuna-dashboard")
        except Exception as e:
            print(f"⚠️  Dashboard startup failed: {e}")
            print("   Continuing without dashboard...")

    @abstractmethod
    def _create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create a model with the given parameters.

        Args:
            params: Parameters to create model with

        Returns:
            Created model instance
        """
        pass

    @abstractmethod
    def _sample_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters using Optuna's trial object.

        Args:
            trial: Optuna trial object for parameter suggestion

        Returns:
            Dictionary of sampled parameters
        """
        pass

    def _compute_score(self, model: BaseEstimator, X: np.ndarray) -> float:
        """Compute clustering quality score.

        Args:
            model: Fitted clustering model
            X: Input data

        Returns:
            Clustering quality score
        """
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )

        # Get labels either from predict() method or labels_ attribute
        if hasattr(model, "predict"):
            labels = model.predict(X)
        elif hasattr(model, "labels_"):
            labels = model.labels_
        else:
            raise ValueError(
                f"Model {type(model).__name__} has neither predict() method nor labels_ attribute"
            )

        # Check if we have valid labels for scoring
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return (
                config.metrics.invalid_score
            )  # Invalid clustering (all points in one cluster)
        if -1 in unique_labels and len(unique_labels) == 2:
            return config.metrics.invalid_score  # Only noise points and one cluster

        # Get the metric function based on configuration
        metric_name = config.metrics.default_metric
        metric_funcs = {
            "silhouette": silhouette_score,
            "calinski_harabasz": calinski_harabasz_score,
            "davies_bouldin": davies_bouldin_score,
        }

        if metric_name not in metric_funcs:
            raise ValueError(f"Unknown metric: {metric_name}")

        return metric_funcs[metric_name](X, labels)

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Score for the trial
        """
        # Sample parameters
        params = self._sample_parameters(trial)

        try:
            # Create and fit model
            model = self._create_model(params)
            model.fit(self.X)

            # Compute score
            score = self._compute_score(model, self.X)

            return score

        except Exception as e:
            # Return worst possible score on failure
            print(f"Trial failed: {str(e)}")
            return float("-inf")

    def optimize(self, X: np.ndarray) -> OptimizationResult:
        """Run optimization process.

        Args:
            X: Input data to optimize clustering for

        Returns:
            OptimizationResult containing best model and optimization history
        """
        self.X = X

        # Set up study and dashboard
        self._setup_study()
        self._start_dashboard()

        start_time = time.time()

        # Run optimization
        callbacks = []
        if self.max_trials:
            callbacks.append(optuna._callbacks.MaxTrialsCallback(self.max_trials))

        self.study.optimize(
            self._objective,
            n_trials=self.max_trials,
            callbacks=callbacks,
        )

        optimization_time = time.time() - start_time

        # Get best results
        best_trial = self.study.best_trial
        self.best_model = self._create_model(best_trial.params)
        self.best_model.fit(self.X)  # Fit the best model before returning

        # Clean up dashboard process
        if self.dashboard_process:
            print(
                f"🌐 Dashboard still running at http://localhost:{self.dashboard_port}"
            )
            print("   (Process will continue running until manually stopped)")

        # Convert trials to history
        history = []
        for trial in self.study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })

        return OptimizationResult(
            best_score=best_trial.value,
            best_params=best_trial.params,
            best_model=self.best_model,
            study=self.study,
            convergence_info={
                "converged": True,
                "total_trials": len(self.study.trials),
                "best_trial": best_trial.number,
            },
            execution_stats={
                "execution_time": optimization_time,
                "trials_per_second": len(self.study.trials) / optimization_time,
            },
            history=history,
        )
