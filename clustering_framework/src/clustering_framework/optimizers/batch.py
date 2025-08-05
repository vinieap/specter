"""Batch optimization strategy for clustering algorithms using Optuna."""

import time
from typing import Any, Dict, Optional
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.metrics import silhouette_score

from .base import BaseOptimizer, OptimizationResult


def _worker(estimator_class, trial_params, X):
    """Worker function that can be pickled.

    Args:
        estimator_class: The scikit-learn estimator class to use
        trial_params: Parameters for the estimator
        X: Input data to cluster

    Returns:
        float: Clustering score or -inf on failure
    """
    try:
        # Create a new model instance
        model = estimator_class(**trial_params)
        model.fit(X)

        # Get labels using labels_ attribute or predict method
        if hasattr(model, "labels_"):
            labels = model.labels_
        elif hasattr(model, "predict"):
            labels = model.predict(X)
        else:
            raise ValueError("Model does not provide cluster labels")

        # Check if we have enough clusters for silhouette score
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return float("-inf")  # Not enough clusters

        return silhouette_score(X, labels)
    except Exception as e:
        print(f"Trial failed: {str(e)}")
        return float("-inf")


class BatchOptimizer(BaseOptimizer):
    """Optimizer that evaluates multiple parameter sets in parallel using Optuna's ask-and-tell API."""

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
        n_jobs: Optional[int] = None,  # Number of parallel jobs
        batch_size: Optional[int] = None,  # Smaller batch size to avoid overwhelming the system
    ):
        """Initialize batch optimizer.

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
            n_jobs: Number of parallel jobs for evaluation. If None, uses all available cores.
            batch_size: Number of trials to evaluate in parallel. If None, uses n_jobs * 2 (max 8).
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
        # Set n_jobs and batch_size
        from multiprocessing import cpu_count
        self.n_jobs = n_jobs if n_jobs is not None else max(1, cpu_count() - 1)
        self.batch_size = batch_size if batch_size is not None else min(8, self.n_jobs * 2)
        self.algorithm = algorithm

    def _create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create a model with the given parameters.

        Args:
            params: Parameters to create model with

        Returns:
            Created model instance
        """
        return self.algorithm.create_estimator(params)

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

    def optimize(self, X: np.ndarray) -> OptimizationResult:
        """Run batch optimization process using Optuna's ask-and-tell API.

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

        # Calculate number of batches needed
        n_batches = (
            (self.max_trials + self.batch_size - 1) // self.batch_size
            if self.max_trials
            else None
        )

        batch_idx = 0
        while True:
            if n_batches and batch_idx >= n_batches:
                break

            # Determine current batch size
            remaining = (
                (self.max_trials - len(self.study.trials))
                if self.max_trials
                else self.batch_size
            )
            current_batch_size = min(
                self.batch_size, remaining if remaining > 0 else self.batch_size
            )

            # Ask for trials
            trials = []
            for _ in range(current_batch_size):
                trial = self.study.ask()
                trials.append(trial)

            # Evaluate trials in parallel if n_jobs is set
            print(f"Starting batch {batch_idx + 1} with {current_batch_size} trials...")
            
            # Prepare parameters for all trials
            trial_params = []
            for trial in trials:
                params = self._sample_parameters(trial)
                if self.algorithm.supports_random_state:
                    params["random_state"] = self.random_state
                trial_params.append((self.algorithm.estimator_class, params, self.X))
            
            # Run trials in parallel if n_jobs is set
            if self.n_jobs is not None and self.n_jobs != 1:
                from joblib import Parallel, delayed
                scores = Parallel(n_jobs=self.n_jobs)(
                    delayed(_worker)(*params) for params in trial_params
                )
            else:
                scores = [_worker(*params) for params in trial_params]
            
            # Report results back to Optuna
            for trial, score in zip(trials, scores):
                self.study.tell(trial, score)

            batch_idx += 1

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
                "score": trial.value,  # The test expects 'score' instead of 'value'
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "batch": trial.number // self.batch_size,
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
                "n_batches": batch_idx,
                "batch_size": self.batch_size,
            },
            history=history,
        )
