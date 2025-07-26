import subprocess
import time

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.cluster import SpectralClustering

from .config import DEFAULT_VERBOSITY, N_CORES, VerbosityLevel
from .evaluation import evaluate_params_worker, parallel_objective_function
from .parameters import prepare_clusterer_params, sample_params
from .utils import format_params_for_display


class BayesianSpectralOptimizer:
    """Bayesian optimization for spectral clustering with multiprocessing support"""

    def __init__(
        self,
        n_calls=100,
        n_jobs=None,
        batch_size=4,
        verbosity=DEFAULT_VERBOSITY,
        random_state=42,
        use_dashboard=False,
        dashboard_port=8081,
    ):
        self.n_calls = n_calls
        self.n_jobs = n_jobs or N_CORES
        self.batch_size = batch_size
        self.verbosity = verbosity
        self.random_state = random_state
        self.use_dashboard = use_dashboard
        self.dashboard_port = dashboard_port
        self.X = None
        self.evaluation_history = []
        self.best_score = -np.inf
        self.best_params = None
        self.current_evaluation = 0
        self.study = None

    def set_data(self, X):
        """Set the dataset for optimization"""
        self.X = X.copy()

        # Create Optuna study with storage if dashboard is enabled
        if self.use_dashboard:
            # Use shared SQLite storage for dashboard access
            storage_name = "sqlite:///optuna_spectral_studies.db"
            # Create storage and ensure tables are initialized
            storage = optuna.storages.RDBStorage(url=storage_name)
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.study = optuna.create_study(
                study_name=f"sequential_spectral_{timestamp}_seed{self.random_state}",
                direction="maximize",  # We want to maximize silhouette score
                sampler=TPESampler(
                    n_startup_trials=min(20, self.n_calls // 4),
                    seed=self.random_state,
                ),
                storage=storage,
                load_if_exists=True,
            )
        else:
            # Create in-memory study
            self.study = optuna.create_study(
                direction="maximize",  # We want to maximize silhouette score
                sampler=TPESampler(
                    n_startup_trials=min(20, self.n_calls // 4),
                    seed=self.random_state,
                ),
            )

    def objective(self, trial):
        """Objective function for Optuna optimization"""
        if self.X is None:
            raise ValueError("Dataset not set. Call set_data() first.")

        # Increment evaluation counter
        self.current_evaluation += 1

        # Sample parameters using the trial object
        params = sample_params(trial)

        # Show progress based on verbosity
        if self.verbosity >= VerbosityLevel.MEDIUM:
            if self.current_evaluation % 10 == 0 or self.current_evaluation == 1:
                print(f"  [{self.current_evaluation:2d}/{self.n_calls}]")

        # Evaluate single parameter set directly (no multiprocessing overhead)
        # Use lower verbosity to avoid cluttered output
        eval_verbosity = min(self.verbosity, VerbosityLevel.MINIMAL)
        score, success, message = evaluate_params_worker(
            (self.X, params, eval_verbosity)
        )

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

        # Return score for maximization (Optuna will handle the direction)
        return score if success else -1.0

    def batch_objective(self, params_list):
        """Evaluate multiple parameters in parallel for batch optimization"""
        if self.X is None:
            raise ValueError("Dataset not set. Call set_data() first.")

        print(f"  Evaluating batch of {len(params_list)} parameter sets...")
        results = parallel_objective_function(self.X, params_list, self.n_jobs)

        batch_scores = []
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
                print(f"    New best: {score:.4f}")

            # Return negative score for minimization
            batch_scores.append(-score if success else 1.0)

        return batch_scores

    def optimize(self):
        """Run Bayesian optimization"""
        if self.study is None:
            raise ValueError("Dataset not set. Call set_data() first.")

        # Start dashboard if requested
        dashboard_process = None
        if self.use_dashboard:
            try:
                storage_url = "sqlite:///optuna_spectral_studies.db"

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
            print(f"Starting Bayesian optimization with {self.n_calls} evaluations...")
            if self.verbosity >= VerbosityLevel.MEDIUM:
                print(f"Dataset shape: {self.X.shape}")

        start_time = time.time()

        # Run Optuna optimization
        self.study.optimize(self.objective, n_trials=self.n_calls)

        optimization_time = time.time() - start_time

        # Get best trial
        best_trial = self.study.best_trial
        self.best_score = best_trial.value
        self.best_params = best_trial.params

        if self.verbosity >= VerbosityLevel.MINIMAL:
            print(f"\n🏁 Bayesian optimization completed in {optimization_time:.2f}s")
            print(f"🏆 Best score: {self.best_score:.4f}")
            if self.verbosity >= VerbosityLevel.MEDIUM:
                clean_best = (
                    format_params_for_display(self.best_params)
                    if self.best_params
                    else "None"
                )
                print(f"⚙️  Best parameters: {clean_best}")
                print(f"📊 Total evaluations: {len(self.evaluation_history)}")

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
        }

    def get_best_clusterer(self):
        """Create a SpectralClustering instance with the best parameters found"""
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        # Prepare parameters for SpectralClustering
        params_clean = prepare_clusterer_params(self.best_params, self.X)

        return SpectralClustering(**params_clean, random_state=42)
