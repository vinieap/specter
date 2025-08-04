"""High-performance Bayesian optimization for clustering algorithms using ask-and-tell API."""
import subprocess
import time
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler

from .algorithms.registry import algorithm_registry
from .config import DEFAULT_VERBOSITY, N_CORES, VerbosityLevel
from .evaluation import parallel_objective_function
from .utils import format_params_for_display


class BatchBayesianClusteringOptimizer:
    """
    High-performance Bayesian optimization for clustering algorithms using ask-and-tell API.
    Evaluates multiple parameter sets in parallel for better performance and exploration.

    Benefits over sequential optimization:
    - Better exploration through batch diversity
    - Full CPU utilization with parallel evaluations
    - More robust to local optima
    - Optional real-time visualization with optuna-dashboard

    Best for:
    - Large datasets (>500 samples)
    - Many evaluations (>100)
    - Systems with multiple cores
    - When clustering evaluation is expensive

    Dashboard Features:
    - Real-time optimization progress tracking
    - Interactive parameter importance plots
    - Study comparison and analysis
    - Web-based interface accessible via browser
    """

    def __init__(
        self,
        algorithm: str = "spectral",
        n_calls: int = 100,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbosity: int = DEFAULT_VERBOSITY,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        n_startup_trials: Optional[int] = None,
        random_state: int = 42,
        use_dashboard: bool = False,
        dashboard_port: int = 8080,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        algorithm : str, default='spectral'
            The clustering algorithm to optimize
        n_calls : int, default=100
            Number of optimization iterations
        n_jobs : int, optional
            Number of parallel jobs (defaults to cpu_count() - 1)
        batch_size : int, optional
            Batch size for parallel evaluation (defaults to n_jobs * 2, max 8)
        verbosity : int, default=DEFAULT_VERBOSITY
            Verbosity level for output
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler
        n_startup_trials : int, optional
            Number of random trials before optimization
        random_state : int, default=42
            Random seed for reproducibility
        use_dashboard : bool, default=False
            Whether to start optuna-dashboard for real-time visualization
        dashboard_port : int, default=8080
            Port for the optuna-dashboard web interface
        """
        self.algorithm = algorithm_registry.get_algorithm(algorithm, random_state)
        self.n_calls = n_calls
        self.n_jobs = n_jobs or N_CORES
        self.batch_size = batch_size or min(self.n_jobs * 2, 8)  # Adaptive batch size
        self.verbosity = verbosity
        self.n_startup_trials = n_startup_trials or min(20, n_calls // 4)
        self.random_state = random_state
        self.use_dashboard = use_dashboard
        self.dashboard_port = dashboard_port

        self.X = None
        self.study = None
        self.evaluation_history = []
        self.best_score = -np.inf
        self.best_params = None
        self.current_evaluation = 0

        # Create sampler if not provided
        if sampler is None:
            self.sampler = TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=self.random_state,
            )
        else:
            self.sampler = sampler

    def set_data(self, X):
        """Set the dataset for optimization"""
        self.X = X.copy()

        # Create Optuna study with storage if dashboard is enabled
        if self.use_dashboard:
            # Use shared SQLite storage for dashboard access
            storage_name = "sqlite:///optuna_clustering_studies.db"
            # Create storage and ensure tables are initialized
            storage = optuna.storages.RDBStorage(url=storage_name)
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.study = optuna.create_study(
                study_name=f"batch_{self.algorithm.name}_{timestamp}_seed{self.random_state}",
                direction="maximize",  # We want to maximize silhouette score
                sampler=self.sampler,
                storage=storage,
                load_if_exists=True,
            )
        else:
            # Create in-memory study
            self.study = optuna.create_study(
                direction="maximize",  # We want to maximize silhouette score
                sampler=self.sampler,
            )

    def _convert_trials_to_params(self, trials):
        """Convert Optuna trials to parameter dictionaries"""
        params_list = []
        for trial in trials:
            params = self.algorithm.sample_parameters(trial)
            params_list.append(params)
        return params_list

    def _evaluate_batch(self, params_list):
        """Evaluate a batch of parameters in parallel"""
        if self.verbosity >= VerbosityLevel.MEDIUM:
            print(
                f"  Batch [{len(self.evaluation_history)+1}-{len(self.evaluation_history)+len(params_list)}]: {len(params_list)} evaluations"
            )

        # Use our existing parallel evaluation function
        results = parallel_objective_function(
            self.X,
            params_list,
            self.algorithm,
            self.n_jobs,
            self.verbosity,
        )

        scores = []
        for i, (score, success, message) in enumerate(results):
            params = params_list[i]

            # Store evaluation history
            self.evaluation_history.append(
                {
                    "params": params.copy(),
                    "score": score,
                    "success": success,
                    "message": message,
                }
            )

            # Update best if this is better
            if success and score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                if self.verbosity >= VerbosityLevel.MINIMAL:
                    clean_params = format_params_for_display(params)
                    print(f"    🎯 NEW BEST: {score:.4f} ← {clean_params}")

            # Return score (Optuna handles direction based on study settings)
            scores.append(score if success else -1.0)

        return scores

    def optimize(self) -> Dict[str, Any]:
        """Run Bayesian optimization using ask-and-tell API"""
        if self.X is None:
            raise ValueError("Dataset not set. Call set_data() first.")

        # Start dashboard if requested
        dashboard_process = None
        if self.use_dashboard:
            try:
                storage_url = "sqlite:///optuna_clustering_studies.db"

                # Wait a moment for database to be fully initialized
                time.sleep(0.1)

                dashboard_process = subprocess.Popen(
                    [
                        "optuna-dashboard",
                        storage_url,
                        "--port",
                        str(self.dashboard_port),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Give dashboard a moment to start
                time.sleep(1.0)

                if self.verbosity >= VerbosityLevel.MINIMAL:
                    print(
                        f"🌐 Optuna Dashboard started at http://localhost:{self.dashboard_port}"
                    )
                    print("   You can monitor optimization progress in real-time!")
            except (ImportError, FileNotFoundError) as e:
                if self.verbosity >= VerbosityLevel.MINIMAL:
                    print(f"⚠️  Could not start optuna-dashboard: {e}")
                    print("   Install with: pip install optuna-dashboard")
            except Exception as e:
                if self.verbosity >= VerbosityLevel.MINIMAL:
                    print(f"⚠️  Dashboard startup failed: {e}")
                    print("   Continuing without dashboard...")

        if self.verbosity >= VerbosityLevel.MINIMAL:
            print(
                f"Starting batch Bayesian optimization for {self.algorithm.name} clustering with {self.n_calls} evaluations..."
            )
            print(f"Batch size: {self.batch_size}, Parallel jobs: {self.n_jobs}")
            if self.verbosity >= VerbosityLevel.MEDIUM:
                print(f"Dataset shape: {self.X.shape}")

        start_time = time.time()

        # Calculate number of batches needed
        n_batches = (self.n_calls + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            remaining_calls = self.n_calls - len(self.evaluation_history)
            current_batch_size = min(self.batch_size, remaining_calls)

            if current_batch_size <= 0:
                break

            if self.verbosity >= VerbosityLevel.MEDIUM:
                print(f"\n--- Batch {batch_idx + 1}/{n_batches} ---")

            # Ask for trials to evaluate
            trials = []
            for _ in range(current_batch_size):
                trial = self.study.ask()
                trials.append(trial)

            params_batch = self._convert_trials_to_params(trials)

            # Evaluate batch in parallel
            scores = self._evaluate_batch(params_batch)

            # Tell study about the results
            for trial, score in zip(trials, scores):
                # We set direction="maximize", so provide the actual score
                self.study.tell(trial, score)

        optimization_time = time.time() - start_time

        # Get best trial
        best_trial = self.study.best_trial
        self.best_score = best_trial.value
        self.best_params = best_trial.params

        if self.verbosity >= VerbosityLevel.MINIMAL:
            print(
                f"\n🏁 Batch Bayesian optimization completed in {optimization_time:.2f}s"
            )
            print(f"🏆 Best score: {self.best_score:.4f}")
            if self.verbosity >= VerbosityLevel.MEDIUM:
                clean_best = (
                    format_params_for_display(self.best_params)
                    if self.best_params
                    else "None"
                )
                print(f"⚙️  Best parameters: {clean_best}")
                print(f"📊 Total evaluations: {len(self.evaluation_history)}")
                print(
                    f"🚀 Avg evaluations/sec: {len(self.evaluation_history)/optimization_time:.1f}"
                )

        # Cleanup dashboard process
        if self.use_dashboard and dashboard_process:
            if self.verbosity >= VerbosityLevel.MINIMAL:
                print(
                    f"🌐 Dashboard still running at http://localhost:{self.dashboard_port}"
                )
                print("   (Process will continue running until manually stopped)")

        return {
            "study": self.study,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "optimization_time": optimization_time,
            "n_evaluations": len(self.evaluation_history),
            "evaluation_history": self.evaluation_history,
            "evaluations_per_second": len(self.evaluation_history) / optimization_time,
        }

    def get_best_clusterer(self):
        """Create a clusterer instance with the best parameters found"""
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        # Prepare parameters for the clusterer
        params_clean = self.algorithm.prepare_parameters(self.best_params, self.X)

        return self.algorithm.create_estimator(params_clean)