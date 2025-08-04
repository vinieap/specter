"""Multi-study optimization strategy for clustering algorithms."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseOptimizer, OptimizationResult

class Study:
    """Represents a single optimization study with specific focus."""
    
    def __init__(
        self,
        algorithm: Any,  # ClusteringAlgorithm
        param_ranges: Dict[str, Any],
        max_trials: int
    ):
        """Initialize study.
        
        Args:
            algorithm: Clustering algorithm to optimize
            param_ranges: Parameter ranges to focus on
            max_trials: Maximum number of trials for this study
        """
        self.algorithm = algorithm
        self.param_ranges = param_ranges
        self.max_trials = max_trials
        self.best_score = float('-inf')
        self.best_params = None
        self.history: List[Dict[str, Any]] = []
        
    def sample_parameters(self) -> Dict[str, Any]:
        """Sample parameters within study's focused ranges.
        
        Returns:
            Dictionary containing sampled parameters
        """
        params = {}
        for param_name, param_range in self.param_ranges.items():
            if isinstance(param_range, (list, tuple)):
                params[param_name] = np.random.choice(param_range)
            elif isinstance(param_range, dict):
                if param_range['type'] == 'int':
                    params[param_name] = np.random.randint(
                        param_range['min'],
                        param_range['max']
                    )
                elif param_range['type'] == 'float':
                    params[param_name] = np.random.uniform(
                        param_range['min'],
                        param_range['max']
                    )
        return params

class MultiStudyOptimizer(BaseOptimizer):
    """Optimizer that runs multiple parallel optimization studies."""
    
    def __init__(
        self,
        algorithm: Any,  # ClusteringAlgorithm
        num_studies: int = 3,
        trials_per_study: int = 50,
        max_parallel: int = 2,
        max_trials: Optional[int] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        min_trials: int = 20
    ):
        """Initialize multi-study optimizer.
        
        Args:
            algorithm: Clustering algorithm to optimize
            num_studies: Number of parallel studies to run
            trials_per_study: Number of trials per study
            max_parallel: Maximum number of parallel studies
            max_trials: Maximum total number of trials
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
        self.num_studies = num_studies
        self.trials_per_study = trials_per_study
        self.max_parallel = max_parallel
        self.studies: List[Study] = []
        
    def _create_studies(self) -> List[Study]:
        """Create optimization studies with different parameter focus.
        
        Returns:
            List of created studies
        """
        studies = []
        param_space = self.algorithm.get_parameter_ranges()
        
        for _ in range(self.num_studies):
            # Randomly select subset of parameters to focus on
            focus_params = {}
            for param_name, param_range in param_space.items():
                if np.random.random() < 0.7:  # 70% chance to include parameter
                    focus_params[param_name] = param_range
                    
            study = Study(
                algorithm=self.algorithm,
                param_ranges=focus_params,
                max_trials=self.trials_per_study
            )
            studies.append(study)
            
        return studies
        
    def optimize(self, X: np.ndarray) -> OptimizationResult:
        """Run multiple optimization studies in parallel.
        
        Args:
            X: Input data to optimize clustering for
            
        Returns:
            OptimizationResult containing best model and optimization history
        """
        self.studies = self._create_studies()
        
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit studies for execution
            future_to_study = {
                executor.submit(self._run_study, study, X): study
                for study in self.studies
            }
            
            # Collect results as they complete
            all_results = []
            for future in as_completed(future_to_study):
                study = future_to_study[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # Update best overall results
                    if result.best_score > self.best_score:
                        self.best_score = result.best_score
                        self.best_params = result.best_params
                        self.best_model = result.best_model
                except Exception as e:
                    print(f"Study failed with error: {e}")
                    
        # Combine results from all studies
        combined_history = []
        for result in all_results:
            combined_history.extend(result.history)
            
        return OptimizationResult(
            best_score=self.best_score,
            best_params=self.best_params,
            best_model=self.best_model,
            history=combined_history,
            convergence_info={
                'num_studies': len(all_results),
                'studies_converged': sum(
                    1 for r in all_results 
                    if r.convergence_info['converged']
                )
            },
            execution_stats={
                'total_trials': sum(
                    len(r.history) for r in all_results
                ),
                'successful_studies': len(all_results)
            }
        )
        
    def _run_study(self, study: Study, X: np.ndarray) -> OptimizationResult:
        """Run a single optimization study.
        
        Args:
            study: Study to run
            X: Input data
            
        Returns:
            OptimizationResult for the study
        """
        for _ in range(study.max_trials):
            params = study.sample_parameters()
            score = self._evaluate_trial(params, X)
            
            if score > study.best_score:
                study.best_score = score
                study.best_params = params
                
            study.history.append({
                'params': params,
                'score': score
            })
            
        return OptimizationResult(
            best_score=study.best_score,
            best_params=study.best_params,
            best_model=self._create_model(study.best_params),
            history=study.history,
            convergence_info={'converged': True},
            execution_stats={'trials': len(study.history)}
        )
        
    def _create_trial(self) -> Dict[str, Any]:
        """Not used in multi-study optimization."""
        raise NotImplementedError(
            "Multi-study optimizer does not use single trial creation"
        )
        
    def _evaluate_trial(self, params: Dict[str, Any], X: np.ndarray) -> float:
        """Evaluate a trial with given parameters.
        
        Args:
            params: Parameters to evaluate
            X: Input data
            
        Returns:
            Score for the trial
        """
        model = self.algorithm.create_estimator(params)
        model.fit(X)
        return self._compute_score(model, X)
        
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