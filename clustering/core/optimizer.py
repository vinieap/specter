"""Base interface for clustering optimizers."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import numpy as np
import optuna
from sklearn.base import BaseEstimator

from .types import OptimizationResult, ConvergenceStatus, ParamDict, ArrayLike
from .algorithm import ClusteringAlgorithm


class ClusteringOptimizer(ABC):
    """Base class for all clustering optimizers.
    
    This abstract base class defines the interface that all clustering
    optimizers must implement. It provides a standard way to:
    - Set up optimization
    - Run optimization
    - Track progress
    - Get results
    
    Each optimizer implementation should inherit from this class and
    implement all abstract methods.
    """
    
    def __init__(
        self,
        algorithm: ClusteringAlgorithm,
        n_calls: int = 100,
        verbosity: int = 1,
        random_state: Optional[int] = None,
        use_dashboard: bool = False,
        dashboard_port: int = 8080,
    ):
        """Initialize the optimizer.
        
        Parameters
        ----------
        algorithm : ClusteringAlgorithm
            The clustering algorithm to optimize
        n_calls : int, default=100
            Number of optimization iterations
        verbosity : int, default=1
            Verbosity level for output
        random_state : int, optional
            Random seed for reproducibility
        use_dashboard : bool, default=False
            Whether to start optuna-dashboard for visualization
        dashboard_port : int, default=8080
            Port for the optuna-dashboard web interface
        """
        self.algorithm = algorithm
        self.n_calls = n_calls
        self.verbosity = verbosity
        self.random_state = random_state
        self.use_dashboard = use_dashboard
        self.dashboard_port = dashboard_port
        
        # Internal state
        self.X: Optional[ArrayLike] = None
        self.study: Optional[optuna.Study] = None
        self.best_score: float = -np.inf
        self.best_params: Optional[ParamDict] = None
        self.evaluation_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def optimize(self, X: ArrayLike) -> OptimizationResult:
        """Run optimization on the input data.
        
        This method should implement the core optimization logic,
        including:
        - Setting up the optimization study
        - Running trials
        - Tracking progress
        - Handling convergence
        - Returning results
        
        Parameters
        ----------
        X : array-like
            Input data to cluster
        
        Returns
        -------
        OptimizationResult
            Results of the optimization
        """
        pass
    
    @abstractmethod
    def get_best_model(self) -> BaseEstimator:
        """Get the best model found during optimization.
        
        Returns
        -------
        BaseEstimator
            The best clustering model
        
        Raises
        ------
        ValueError
            If optimize() hasn't been called yet
        """
        pass
    
    def check_convergence(self) -> ConvergenceStatus:
        """Check if optimization has converged.
        
        This method analyzes the optimization history to determine
        if the process has converged to a stable solution.
        
        Returns
        -------
        ConvergenceStatus
            Current convergence status
        """
        # Default implementation - subclasses should provide more sophisticated logic
        if len(self.evaluation_history) < 20:
            return ConvergenceStatus(
                converged=False,
                confidence=0.0,
                method="insufficient_data",
                details={
                    "n_evaluations": len(self.evaluation_history),
                    "min_required": 20
                },
                recommendation="Continue optimization to gather more data"
            )
        
        recent_scores = [
            trial["score"] for trial in self.evaluation_history[-10:]
            if trial.get("success", False)
        ]
        
        if not recent_scores:
            return ConvergenceStatus(
                converged=False,
                confidence=0.0,
                method="no_successful_trials",
                details={"recent_trials": 10},
                recommendation="Check algorithm parameters and data"
            )
        
        score_std = np.std(recent_scores)
        score_range = np.max(recent_scores) - np.min(recent_scores)
        
        if score_std < 0.01 and score_range < 0.05:
            return ConvergenceStatus(
                converged=True,
                confidence=1.0 - max(score_std * 100, score_range * 20),
                method="score_stability",
                details={
                    "score_std": score_std,
                    "score_range": score_range
                },
                recommendation="Optimization has likely converged"
            )
        
        return ConvergenceStatus(
            converged=False,
            confidence=0.0,
            method="ongoing",
            details={
                "score_std": score_std,
                "score_range": score_range
            },
            recommendation="Continue optimization"
        )
    
    def _validate_input(self, X: ArrayLike) -> None:
        """Validate input data.
        
        Parameters
        ----------
        X : array-like
            Input data to validate
        
        Raises
        ------
        ValueError
            If input is invalid
        """
        if not isinstance(X, (np.ndarray, list)):
            raise ValueError("Input X must be a numpy array or list")
        
        if isinstance(X, list):
            X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError("Input X must be 2-dimensional")
        
        if X.shape[0] < 2:
            raise ValueError("Input X must have at least 2 samples")
        
        if X.shape[1] < 1:
            raise ValueError("Input X must have at least 1 feature")
    
    def __str__(self) -> str:
        """String representation of the optimizer.
        
        Returns
        -------
        str
            Optimizer description
        """
        return (
            f"{self.__class__.__name__}("
            f"algorithm='{self.algorithm.name}', "
            f"n_calls={self.n_calls})"
        )