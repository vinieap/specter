"""Sequential optimization strategy for clustering algorithms."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator

from .base import BaseOptimizer

class SequentialOptimizer(BaseOptimizer):
    """Optimizer that evaluates parameter sets sequentially with adaptive sampling."""
    
    def __init__(
        self,
        algorithm: Any,  # ClusteringAlgorithm
        exploration_ratio: float = 0.2,
        max_trials: Optional[int] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        min_trials: int = 20
    ):
        """Initialize sequential optimizer.
        
        Args:
            algorithm: Clustering algorithm to optimize
            exploration_ratio: Ratio of trials that should explore new parameter spaces
            max_trials: Maximum number of optimization trials
            patience: Number of trials without improvement before convergence
            min_delta: Minimum change in score to be considered an improvement
            min_trials: Minimum number of trials before allowing convergence
        """
        super().__init__(
            max_trials=max_trials,
            patience=patience,
            min_delta=min_delta,
            min_trials=min_trials
        )
        self.algorithm = algorithm
        self.exploration_ratio = exploration_ratio
        self.parameter_history: List[Dict[str, Any]] = []
        self.score_history: List[float] = []
        
    def _create_trial(self) -> Dict[str, Any]:
        """Create a new trial with adaptively sampled parameters.
        
        Returns:
            Dictionary containing trial parameters
        """
        if (len(self.parameter_history) == 0 or 
            np.random.random() < self.exploration_ratio):
            # Explore new parameter space
            return self.algorithm.sample_parameters()
        else:
            # Exploit successful parameter regions
            return self._sample_from_history()
        
    def _evaluate_trial(self, params: Dict[str, Any], X: np.ndarray) -> float:
        """Evaluate a trial with given parameters.
        
        Args:
            params: Parameters to evaluate
            X: Input data
            
        Returns:
            Score for the trial
        """
        # Create and fit model
        model = self.algorithm.create_estimator(params)
        model.fit(X)
        
        # Compute and store score
        score = self._compute_score(model, X)
        self.parameter_history.append(params)
        self.score_history.append(score)
        
        return score
        
    def _create_model(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create a model with the given parameters.
        
        Args:
            params: Parameters to create model with
            
        Returns:
            Created model instance
        """
        return self.algorithm.create_estimator(params)
        
    def _compute_score(self, model: BaseEstimator, X: np.ndarray) -> float:
        """Compute clustering quality score.
        
        Args:
            model: Fitted clustering model
            X: Input data
            
        Returns:
            Clustering quality score
        """
        from sklearn.metrics import silhouette_score
        labels = model.predict(X)
        return silhouette_score(X, labels)
        
    def _sample_from_history(self) -> Dict[str, Any]:
        """Sample parameters from historical successful trials.
        
        Returns:
            Dictionary containing sampled parameters
        """
        # Weight trials by their scores
        scores = np.array(self.score_history)
        weights = np.exp(scores - np.max(scores))  # Softmax-like weighting
        weights /= np.sum(weights)
        
        # Sample a historical trial
        selected_idx = np.random.choice(len(self.parameter_history), p=weights)
        base_params = self.parameter_history[selected_idx].copy()
        
        # Add small random perturbations
        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                base_params[key] = value * (1 + np.random.normal(0, 0.1))
                
        return base_params