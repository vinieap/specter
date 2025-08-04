"""Progress reporting utilities for clustering optimization."""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import numpy as np
from tqdm.auto import tqdm
import optuna
from optuna.importance import get_param_importances
import matplotlib.pyplot as plt

from .convergence import ConvergenceDetector


class OptimizationProgress:
    """Track and report optimization progress with detailed metrics."""

    def __init__(
        self,
        n_calls: int,
        algorithm_name: str,
        verbosity: int,
        batch_mode: bool = True,
    ):
        """Initialize progress tracking.

        Parameters
        ----------
        n_calls : int
            Total number of optimization calls planned
        algorithm_name : str
            Name of the clustering algorithm
        verbosity : int
            Verbosity level for output
        batch_mode : bool, default=True
            Whether optimization is running in batch mode
        """
        self.n_calls = n_calls
        self.algorithm_name = algorithm_name
        self.verbosity = verbosity
        self.batch_mode = batch_mode
        
        # Initialize tracking variables
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.evaluation_history = []
        self.best_score = -np.inf
        self.best_params = None
        self.n_failed = 0
        self.progress_bar = None
        
        # Performance tracking
        self.evaluation_times: List[float] = []
        self.scores_history: List[float] = []
        
        # Initialize convergence detector
        self.convergence_detector = ConvergenceDetector(
            window_size=10,
            min_evaluations=20,
            significance_level=0.05,
            improvement_threshold=0.01,
        )
        self.last_convergence_check = None
        self.convergence_check_frequency = max(1, n_calls // 20)  # Check every 5%
        
        # Initialize progress bar if needed
        if self.verbosity >= 2:
            self.progress_bar = tqdm(
                total=n_calls,
                desc=f"Optimizing {algorithm_name}",
                unit="eval",
            )

    def update(
        self,
        score: float,
        params: Dict[str, Any],
        success: bool,
        message: str,
        batch_size: Optional[int] = None,
    ):
        """Update progress with new evaluation results.

        Parameters
        ----------
        score : float
            Score from the latest evaluation
        params : dict
            Parameters used in the evaluation
        success : bool
            Whether the evaluation was successful
        message : str
            Any message from the evaluation
        batch_size : int, optional
            Size of the current batch (for batch mode)
        """
        current_time = time.time()
        evaluation_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Store evaluation data
        self.evaluation_history.append({
            "params": params.copy(),
            "score": score,
            "success": success,
            "message": message,
            "time": evaluation_time,
        })
        
        if success:
            self.scores_history.append(score)
            self.evaluation_times.append(evaluation_time)
            
            # Update best score
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self._report_new_best()
        else:
            self.n_failed += 1
        
        # Update progress bar
        if self.progress_bar is not None:
            increment = batch_size if batch_size else 1
            self.progress_bar.update(increment)
            
            # Update progress bar description with current metrics
            desc = self._get_progress_description()
            self.progress_bar.set_description(desc)
        
        # Check convergence periodically
        n_evals = len(self.evaluation_history)
        if n_evals % self.convergence_check_frequency == 0:
            self._check_and_report_convergence()
        
        # Detailed reporting based on verbosity
        self._report_progress()

    def _get_progress_description(self) -> str:
        """Get current progress description for progress bar."""
        n_evals = len(self.evaluation_history)
        success_rate = (n_evals - self.n_failed) / max(1, n_evals) * 100
        
        desc = f"{self.algorithm_name}: best={self.best_score:.4f}, success={success_rate:.1f}%"
        
        # Add convergence status if available
        if self.last_convergence_check and self.last_convergence_check.converged:
            desc += " [Converging]"
        
        return desc

    def _check_and_report_convergence(self):
        """Check convergence and report status."""
        self.last_convergence_check = self.convergence_detector.check_convergence(
            scores=self.scores_history,
            times=self.evaluation_times,
            n_total=self.n_calls,
            best_score=self.best_score,
        )
        
        if self.verbosity >= 2:
            if self.last_convergence_check.converged:
                print(f"\n🎯 Convergence detected ({self.last_convergence_check.method}):")
                print(f"  Confidence: {self.last_convergence_check.confidence:.2f}")
                print(f"  Recommendation: {self.last_convergence_check.recommendation}")
                if self.verbosity >= 3:
                    print("  Details:")
                    for key, value in self.last_convergence_check.details.items():
                        print(f"    {key}: {value}")

    def _report_new_best(self):
        """Report when a new best score is found."""
        if self.verbosity >= 1:
            print(f"\n🎯 NEW BEST: {self.best_score:.4f}")
            if self.verbosity >= 2:
                print("Parameters:")
                for key, value in sorted(self.best_params.items()):
                    print(f"  {key}: {value}")

    def _report_progress(self):
        """Report detailed progress based on verbosity level."""
        if self.verbosity < 2:
            return
            
        n_evals = len(self.evaluation_history)
        
        # Report every 10% progress or when verbosity is high
        if n_evals % max(1, self.n_calls // 10) == 0 or self.verbosity >= 3:
            self._print_current_metrics()

    def _print_current_metrics(self):
        """Print current optimization metrics."""
        n_evals = len(self.evaluation_history)
        elapsed_time = time.time() - self.start_time
        
        # Calculate metrics
        success_rate = (n_evals - self.n_failed) / max(1, n_evals) * 100
        avg_time = np.mean(self.evaluation_times) if self.evaluation_times else 0
        evals_per_sec = n_evals / elapsed_time
        
        # Estimate remaining time
        remaining_evals = self.n_calls - n_evals
        estimated_remaining = remaining_evals / evals_per_sec
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        
        # Print report
        print(f"\n📊 Progress Report:")
        print(f"  Evaluations: {n_evals}/{self.n_calls}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Average evaluation time: {avg_time:.3f}s")
        print(f"  Evaluations/second: {evals_per_sec:.1f}")
        print(f"  Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        print(f"  Estimated completion: {eta.strftime('%H:%M:%S')}")
        
        # Add convergence status if available
        if self.last_convergence_check:
            status = "Converging" if self.last_convergence_check.converged else "Exploring"
            print(f"  Convergence status: {status}")
            if self.last_convergence_check.converged:
                print(f"  Convergence confidence: {self.last_convergence_check.confidence:.2f}")

    def get_summary(self, study: optuna.Study) -> Dict[str, Any]:
        """Get a summary of the optimization process.

        Parameters
        ----------
        study : optuna.Study
            The completed Optuna study

        Returns
        -------
        dict
            Summary statistics and visualizations
        """
        elapsed_time = time.time() - self.start_time
        n_evals = len(self.evaluation_history)
        
        # Calculate parameter importances
        try:
            param_importances = get_param_importances(study)
        except Exception:
            param_importances = {}
        
        # Create parameter importance plot
        if param_importances and self.verbosity >= 2:
            plt.figure(figsize=(10, 5))
            importance_items = sorted(
                param_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            params, importances = zip(*importance_items)
            plt.barh(params, importances)
            plt.title("Parameter Importances")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            
            # Save plot for return
            import io
            import base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        else:
            plot_data = None
        
        # Get final convergence status
        final_convergence = self.convergence_detector.check_convergence(
            self.scores_history,
            self.evaluation_times,
            self.n_calls,
            self.best_score,
        )
        
        return {
            "n_evaluations": n_evals,
            "n_failed": self.n_failed,
            "success_rate": (n_evals - self.n_failed) / max(1, n_evals) * 100,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "total_time": elapsed_time,
            "average_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0,
            "evaluations_per_second": n_evals / elapsed_time,
            "param_importances": param_importances,
            "importance_plot": plot_data,
            "scores_history": self.scores_history.copy(),
            "evaluation_times": self.evaluation_times.copy(),
            "convergence_status": {
                "converged": final_convergence.converged,
                "confidence": final_convergence.confidence,
                "method": final_convergence.method,
                "recommendation": final_convergence.recommendation,
                "details": final_convergence.details,
            },
        }

    def close(self):
        """Clean up progress tracking resources."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None