"""Multi-study analysis for clustering optimization."""
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from .api import optimize_clustering
from .algorithms.registry import algorithm_registry


@dataclass
class StudyResult:
    """Results from a single optimization study."""
    algorithm: str
    seed: int
    best_score: float
    best_params: Dict[str, Any]
    optimization_time: float
    n_evaluations: int
    evaluations_per_second: float
    convergence_status: Dict[str, Any]
    evaluation_history: List[Dict[str, Any]]


@dataclass
class AlgorithmPerformance:
    """Aggregated performance metrics for an algorithm across multiple seeds."""
    algorithm: str
    mean_score: float
    std_score: float
    best_score: float
    worst_score: float
    mean_time: float
    mean_evaluations: float
    best_params: Dict[str, Any]  # Parameters that achieved the best score
    success_rate: float
    parameter_importance: Dict[str, float]
    convergence_rate: float
    score_stability: float  # Standard deviation relative to mean


class MultiStudyOptimizer:
    """Runs multiple optimization studies with different seeds and algorithms."""

    def __init__(
        self,
        n_seeds: int = 5,
        algorithms: Optional[List[str]] = None,
        n_calls: int = 100,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbosity: int = 1,
        use_batch_optimizer: bool = True,
    ):
        """Initialize multi-study optimizer.

        Parameters
        ----------
        n_seeds : int, default=5
            Number of different random seeds to use
        algorithms : List[str], optional
            List of algorithms to evaluate. If None, uses all available algorithms.
        n_calls : int, default=100
            Number of optimization iterations per study
        n_jobs : int, optional
            Number of parallel jobs
        batch_size : int, optional
            Batch size for parallel evaluation
        verbosity : int, default=1
            Verbosity level
        use_batch_optimizer : bool, default=True
            Whether to use batch optimization
        """
        self.n_seeds = n_seeds
        self.algorithms = algorithms or algorithm_registry.list_algorithms()
        self.n_calls = n_calls
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbosity = verbosity
        self.use_batch_optimizer = use_batch_optimizer
        
        # Generate random seeds if not using a fixed seed
        self.seeds = [random.randint(0, 100000) for _ in range(n_seeds)]
        
        self.results: Dict[str, Dict[int, StudyResult]] = {}
        self.performance_metrics: Dict[str, AlgorithmPerformance] = {}

    def optimize(self, X: np.ndarray) -> Dict[str, AlgorithmPerformance]:
        """Run optimization studies for all algorithms and seeds.

        Parameters
        ----------
        X : array-like
            Input data to cluster

        Returns
        -------
        Dict[str, AlgorithmPerformance]
            Performance metrics for each algorithm
        """
        total_studies = len(self.algorithms) * self.n_seeds
        current_study = 0

        for algorithm in self.algorithms:
            self.results[algorithm] = {}
            
            if self.verbosity >= 1:
                print(f"\nOptimizing {algorithm} clustering...")
            
            for seed in self.seeds:
                current_study += 1
                if self.verbosity >= 1:
                    print(f"\nStudy {current_study}/{total_studies}")
                    print(f"Algorithm: {algorithm}, Seed: {seed}")

                # Run optimization study
                results = optimize_clustering(
                    X,
                    algorithm=algorithm,
                    n_calls=self.n_calls,
                    n_jobs=self.n_jobs,
                    batch_size=self.batch_size,
                    verbosity=max(0, self.verbosity - 1),
                    use_batch_optimizer=self.use_batch_optimizer,
                    random_state=seed,
                )

                # Store results
                self.results[algorithm][seed] = StudyResult(
                    algorithm=algorithm,
                    seed=seed,
                    best_score=results["best_score"],
                    best_params=results["best_params"],
                    optimization_time=results["optimization_time"],
                    n_evaluations=results["n_evaluations"],
                    evaluations_per_second=results["evaluations_per_second"],
                    convergence_status=results.get("convergence_status", {}),
                    evaluation_history=results["evaluation_history"],
                )

        # Compute aggregated performance metrics
        self._compute_performance_metrics()
        
        if self.verbosity >= 1:
            self._print_summary()
        
        return self.performance_metrics

    def _compute_performance_metrics(self):
        """Compute aggregated performance metrics for each algorithm."""
        for algorithm in self.algorithms:
            studies = self.results[algorithm].values()
            scores = [study.best_score for study in studies]
            times = [study.optimization_time for study in studies]
            evals = [study.n_evaluations for study in studies]
            
            # Find study with best score
            best_study = max(studies, key=lambda x: x.best_score)
            
            # Calculate success rate (studies that found valid clustering)
            success_rate = sum(1 for s in scores if s > 0) / len(scores)
            
            # Calculate convergence rate
            convergence_rate = sum(
                1 for s in studies if s.convergence_status.get("converged", False)
            ) / len(studies)
            
            # Calculate parameter importance
            param_importance = self._calculate_parameter_importance(algorithm)
            
            # Calculate score stability
            score_stability = np.std(scores) / (np.mean(scores) + 1e-10)
            
            self.performance_metrics[algorithm] = AlgorithmPerformance(
                algorithm=algorithm,
                mean_score=np.mean(scores),
                std_score=np.std(scores),
                best_score=max(scores),
                worst_score=min(scores),
                mean_time=np.mean(times),
                mean_evaluations=np.mean(evals),
                best_params=best_study.best_params,
                success_rate=success_rate,
                parameter_importance=param_importance,
                convergence_rate=convergence_rate,
                score_stability=score_stability,
            )

    def _calculate_parameter_importance(self, algorithm: str) -> Dict[str, float]:
        """Calculate parameter importance across all studies for an algorithm."""
        # Collect all parameter values and corresponding scores
        param_values = []
        scores = []
        
        for study in self.results[algorithm].values():
            for eval_result in study.evaluation_history:
                if eval_result["success"]:
                    param_values.append(eval_result["params"])
                    scores.append(eval_result["score"])
        
        if not param_values:
            return {}
            
        # Convert to DataFrame
        df = pd.DataFrame(param_values)
        df["score"] = scores
        
        # Calculate correlation between parameters and scores
        importance = {}
        for column in df.columns:
            if column != "score" and df[column].dtype in [np.float64, np.int64]:
                correlation = df[column].corr(df["score"])
                importance[column] = abs(correlation)
        
        # Normalize importance scores
        total = sum(importance.values()) + 1e-10
        return {k: v/total for k, v in importance.items()}

    def _print_summary(self):
        """Print summary of all optimization results."""
        print("\n=== Multi-Study Optimization Summary ===")
        print(f"Number of seeds: {self.n_seeds}")
        print(f"Number of algorithms: {len(self.algorithms)}")
        print(f"Total studies: {len(self.algorithms) * self.n_seeds}")
        
        # Print table of results
        headers = ["Algorithm", "Mean Score", "Std Dev", "Best Score", "Success Rate", "Conv. Rate"]
        row_format = "{:<15} {:<10} {:<10} {:<10} {:<12} {:<10}"
        
        print("\n" + row_format.format(*headers))
        print("-" * 70)
        
        for algorithm, metrics in self.performance_metrics.items():
            row = [
                algorithm[:15],
                f"{metrics.mean_score:.4f}",
                f"{metrics.std_score:.4f}",
                f"{metrics.best_score:.4f}",
                f"{metrics.success_rate:.2f}",
                f"{metrics.convergence_rate:.2f}",
            ]
            print(row_format.format(*row))
        
        # Print best performing algorithm
        best_algo = max(
            self.performance_metrics.items(),
            key=lambda x: x[1].mean_score
        )[0]
        
        print(f"\nBest performing algorithm: {best_algo}")
        metrics = self.performance_metrics[best_algo]
        print(f"Mean score: {metrics.mean_score:.4f} ± {metrics.std_score:.4f}")
        print(f"Best parameters: {metrics.best_params}")
        
        # Print parameter importance for best algorithm
        print("\nParameter importance:")
        for param, importance in sorted(
            metrics.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"  {param}: {importance:.3f}")