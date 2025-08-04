"""Base optimizer interface and common optimization logic."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Protocol
import numpy as np
from tqdm import tqdm

@dataclass
class OptimizationResult:
    """Structured optimization results.
    
    Attributes:
        best_score: The best score achieved during optimization
        best_params: Parameters that achieved the best score
        best_model: The best performing model
        history: List of all trials and their results
        convergence_info: Information about optimization convergence
        execution_stats: Statistics about the optimization process
    """
    best_score: float
    best_params: Dict[str, Any]
    best_model: Any  # BaseEstimator from sklearn
    history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_stats: Dict[str, Any]

class ConvergenceDetector:
    """Detects convergence in optimization process."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        min_trials: int = 20
    ):
        """Initialize convergence detector.
        
        Args:
            patience: Number of trials without improvement before convergence
            min_delta: Minimum change in score to be considered an improvement
            min_trials: Minimum number of trials before allowing convergence
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_trials = min_trials
        self.best_score = float('-inf')
        self.trials_without_improvement = 0
        self.trial_count = 0
        
    def update(self, score: float) -> bool:
        """Update convergence state with new score.
        
        Args:
            score: The score from the latest trial
            
        Returns:
            bool: True if converged, False otherwise
        """
        self.trial_count += 1
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1
            
        if self.trial_count < self.min_trials:
            return False
            
        return self.trials_without_improvement >= self.patience

class ProgressTracker:
    """Tracks and reports optimization progress."""
    
    def __init__(self, total_trials: Optional[int] = None):
        """Initialize progress tracker.
        
        Args:
            total_trials: Expected total number of trials (if known)
        """
        self.total_trials = total_trials
        self.current_trial = 0
        self.best_score = float('-inf')
        self.history: List[Dict[str, Any]] = []
        self.pbar = tqdm(total=total_trials, desc="Optimizing")
        
    def update(self, trial_result: Dict[str, Any]):
        """Update progress with new trial result.
        
        Args:
            trial_result: Dictionary containing trial results
        """
        self.current_trial += 1
        self.history.append(trial_result)
        
        score = trial_result.get('score', float('-inf'))
        if score > self.best_score:
            self.best_score = score
            self.pbar.set_postfix(best_score=f"{score:.4f}")
            
        self.pbar.update(1)
        
    def close(self):
        """Clean up progress tracking resources."""
        self.pbar.close()

class BaseOptimizer(ABC):
    """Base class for optimization strategies."""
    
    def __init__(
        self,
        max_trials: Optional[int] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        min_trials: int = 20
    ):
        """Initialize base optimizer.
        
        Args:
            max_trials: Maximum number of optimization trials
            patience: Number of trials without improvement before convergence
            min_delta: Minimum change in score to be considered an improvement
            min_trials: Minimum number of trials before allowing convergence
        """
        self.max_trials = max_trials
        self.convergence_detector = ConvergenceDetector(
            patience=patience,
            min_delta=min_delta,
            min_trials=min_trials
        )
        self.progress_tracker = None
        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = {}
        
    @abstractmethod
    def _create_trial(self) -> Dict[str, Any]:
        """Create a new trial with parameters to evaluate.
        
        Returns:
            Dictionary containing trial parameters
        """
        pass
        
    @abstractmethod
    def _evaluate_trial(self, params: Dict[str, Any], X: np.ndarray) -> float:
        """Evaluate a trial with given parameters.
        
        Args:
            params: Parameters to evaluate
            X: Input data
            
        Returns:
            Score for the trial
        """
        pass
        
    def optimize(self, X: np.ndarray) -> OptimizationResult:
        """Run optimization process.
        
        Args:
            X: Input data to optimize clustering for
            
        Returns:
            OptimizationResult containing best model and optimization history
        """
        self.progress_tracker = ProgressTracker(self.max_trials)
        start_time = time.time()
        
        while (self.max_trials is None or 
               self.progress_tracker.current_trial < self.max_trials):
            
            # Create and evaluate trial
            params = self._create_trial()
            score = self._evaluate_trial(params, X)
            
            # Update tracking
            trial_result = {
                'trial_num': self.progress_tracker.current_trial + 1,
                'params': params,
                'score': score
            }
            self.progress_tracker.update(trial_result)
            
            # Update best results
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_model = self._create_model(params)
                
            # Check convergence
            if self.convergence_detector.update(score):
                break
                
        # Clean up and create result
        self.progress_tracker.close()
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            best_score=self.best_score,
            best_params=self.best_params,
            best_model=self.best_model,
            history=self.progress_tracker.history,
            convergence_info={
                'converged': True,
                'total_trials': self.progress_tracker.current_trial,
                'trials_without_improvement': 
                    self.convergence_detector.trials_without_improvement
            },
            execution_stats={
                'execution_time': execution_time,
                'trials_per_second': 
                    self.progress_tracker.current_trial / execution_time
            }
        )
        
    @abstractmethod
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """Create a model with the given parameters.
        
        Args:
            params: Parameters to create model with
            
        Returns:
            Created model instance
        """
        pass