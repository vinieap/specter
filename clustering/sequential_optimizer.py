"""Sequential Bayesian optimization for clustering algorithms."""
import time
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.exceptions import ConvergenceWarning
import warnings

from .algorithms.registry import algorithm_registry
from .config import DEFAULT_VERBOSITY, VerbosityLevel
from .evaluation import evaluate_clustering
from .progress import OptimizationProgress
from .utils import format_params_for_display


class SequentialClusteringOptimizer:
    """
    Sequential Bayesian optimization for clustering algorithms.
    Evaluates parameter sets one at a time, which can be preferable for:
    - Small datasets
    - Quick evaluations
    - Memory-constrained environments
    - Debugging and development

    Features:
    - Lower memory usage than batch optimization
    - Simpler execution flow
    - Real-time feedback for each evaluation
    - Optional dashboard integration
    """

    def __init__(
        self,
        algorithm: str = "spectral",
        n_calls: int = 100,
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
        self.progress = None

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
                study_name=f"sequential_{self.algorithm.name}_{timestamp}_seed{self.random_state}",
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

    def _objective(self, trial):
        """Objective function for optimization."""
        # Sample parameters
        params = self.algorithm.sample_parameters(trial)

        # Evaluate clustering
        score, success, message = evaluate_clustering(self.X, params, self.algorithm)

        # Update progress
        if self.progress is not None:
            self.progress.update(score, params, success, message)

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

        return score if success else -1.0

    def optimize(self) -> Dict[str, Any]:
        """Run sequential Bayesian optimization"""
        if self.X is None:
            raise ValueError("Dataset not set. Call set_data() first.")

        # Start dashboard if requested
        if self.use_dashboard:
            try:
                import subprocess
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

        # Initialize progress tracking
        self.progress = OptimizationProgress(
            n_calls=self.n_calls,
            algorithm_name=self.algorithm.name,
            verbosity=self.verbosity,
            batch_mode=False,
        )

        if self.verbosity >= VerbosityLevel.MINIMAL:
            print(
                f"Starting sequential Bayesian optimization for {self.algorithm.name} clustering with {self.n_calls} evaluations..."
            )
            if self.verbosity >= VerbosityLevel.MEDIUM:
                print(f"Dataset shape: {self.X.shape}")

        # Suppress convergence warnings during optimization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # Run optimization
            self.study.optimize(
                self._objective,
                n_trials=self.n_calls,
                show_progress_bar=False,  # We use our own progress tracking
            )

        # Get optimization summary
        summary = self.progress.get_summary(self.study)
        
        # Get best trial
        best_trial = self.study.best_trial
        self.best_score = best_trial.value
        self.best_params = best_trial.params

        # Cleanup
        self.progress.close()
        self.progress = None

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
            "optimization_time": summary["total_time"],
            "n_evaluations": summary["n_evaluations"],
            "evaluation_history": self.evaluation_history,
            "evaluations_per_second": summary["evaluations_per_second"],
            "param_importances": summary["param_importances"],
            "importance_plot": summary["importance_plot"],
            "scores_history": summary["scores_history"],
            "evaluation_times": summary["evaluation_times"],
        }

    def get_best_clusterer(self):
        """Create a clusterer instance with the best parameters found"""
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        # Prepare parameters for the clusterer
        params_clean = self.algorithm.prepare_parameters(self.best_params, self.X)

        return self.algorithm.create_estimator(params_clean)