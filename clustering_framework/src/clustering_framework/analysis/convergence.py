"""Convergence analysis module for clustering algorithms.

This module provides tools for analyzing the convergence behavior of clustering algorithms,
including stability metrics and convergence detection.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import adjusted_rand_score, silhouette_score

@dataclass
class ConvergenceAnalysis:
    """Structured convergence analysis results.
    
    Attributes:
        converged: Whether the algorithm has converged
        n_iterations: Number of iterations until convergence
        stability_score: Measure of clustering stability
        convergence_curve: History of convergence metrics
        recommendations: List of recommendations for improving convergence
    """
    converged: bool
    n_iterations: int
    stability_score: float
    convergence_curve: Dict[str, np.ndarray]
    recommendations: List[str]

class ConvergenceAnalyzer:
    """Analyzes convergence behavior of clustering algorithms."""
    
    def __init__(self, 
                 stability_threshold: float = 0.95,
                 max_iterations: int = 100,
                 n_stability_runs: int = 5):
        """Initialize the convergence analyzer.
        
        Args:
            stability_threshold: Threshold for considering clusters stable
            max_iterations: Maximum number of iterations to analyze
            n_stability_runs: Number of runs for stability analysis
        """
        self.stability_threshold = stability_threshold
        self.max_iterations = max_iterations
        self.n_stability_runs = n_stability_runs
    
    def analyze(self, 
                estimator: BaseEstimator,
                X: np.ndarray,
                iteration_callback = None) -> ConvergenceAnalysis:
        """Analyze convergence behavior of a clustering algorithm.
        
        Args:
            estimator: Clustering estimator to analyze
            X: Input data array
            iteration_callback: Optional callback for each iteration
            
        Returns:
            ConvergenceAnalysis object containing analysis results
        """
        # Track convergence metrics
        inertias = []
        silhouettes = []
        labels_history = []
        
        # Run clustering multiple times for stability analysis
        stability_scores = []
        for _ in range(self.n_stability_runs):
            labels = None
            for i in range(self.max_iterations):
                prev_labels = labels
                
                # Fit the estimator
                estimator.fit(X)
                labels = estimator.labels_
                
                # Calculate metrics
                if hasattr(estimator, 'inertia_'):
                    inertias.append(estimator.inertia_)
                if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
                    silhouettes.append(silhouette_score(X, labels))
                labels_history.append(labels)
                
                # Check convergence
                if prev_labels is not None:
                    stability = adjusted_rand_score(prev_labels, labels)
                    stability_scores.append(stability)
                    
                    if stability >= self.stability_threshold:
                        break
                        
                if iteration_callback:
                    iteration_callback(i, labels)
                    
        # Analyze results
        converged = len(stability_scores) > 0 and np.mean(stability_scores[-self.n_stability_runs:]) >= self.stability_threshold
        n_iterations = len(labels_history)
        mean_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        convergence_curve = {
            'inertia': np.array(inertias) if inertias else np.array([]),
            'silhouette': np.array(silhouettes) if silhouettes else np.array([]),
            'stability': np.array(stability_scores) if stability_scores else np.array([])
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            converged, n_iterations, mean_stability, convergence_curve
        )
        
        return ConvergenceAnalysis(
            converged=converged,
            n_iterations=n_iterations,
            stability_score=mean_stability,
            convergence_curve=convergence_curve,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                converged: bool,
                                n_iterations: int,
                                stability_score: float,
                                convergence_curve: Dict[str, np.ndarray]) -> List[str]:
        """Generate recommendations based on convergence analysis."""
        recommendations = []
        
        if not converged:
            recommendations.extend([
                "Algorithm did not converge within maximum iterations",
                "Consider increasing max_iterations",
                "Try different initialization methods"
            ])
            
        if stability_score < 0.8:
            recommendations.extend([
                "Clustering results are not stable",
                "Consider using a different algorithm",
                "Try adjusting algorithm parameters"
            ])
            
        if n_iterations >= self.max_iterations:
            recommendations.extend([
                "Maximum iterations reached",
                "Consider increasing max_iterations",
                "Check if data needs preprocessing"
            ])
            
        # Analyze convergence curve
        if len(convergence_curve['inertia']) > 0:
            inertia_change = np.diff(convergence_curve['inertia'])
            if np.any(inertia_change > 0):
                recommendations.append(
                    "Non-monotonic convergence detected - consider adjusting learning rate"
                )
                
        if len(convergence_curve['silhouette']) > 0:
            if np.mean(convergence_curve['silhouette']) < 0.5:
                recommendations.extend([
                    "Low silhouette scores indicate poor cluster separation",
                    "Try different numbers of clusters",
                    "Consider feature engineering or dimensionality reduction"
                ])
                
        return recommendations


def analyze_convergence(model: BaseEstimator, X: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Analyze convergence behavior of a clustering model.

    This function analyzes the convergence behavior of a fitted clustering model,
    including stability metrics and convergence detection.

    Parameters
    ----------
    model : BaseEstimator
        The fitted clustering model to analyze
    X : array-like, optional
        Input data used for clustering. Required for stability analysis.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing convergence analysis results:
        - converged: bool, whether the algorithm converged
        - n_iterations: int, number of iterations until convergence
        - stability_score: float, measure of clustering stability
        - recommendations: List[str], suggestions for improvement

    Raises
    ------
    ValueError
        If X is not provided but required for analysis
    """
    if X is None:
        raise ValueError("Input data X is required for convergence analysis")

    analyzer = ConvergenceAnalyzer()
    results = analyzer.analyze(model, X)
    
    return {
        "converged": results.converged,
        "n_iterations": results.n_iterations,
        "stability_score": results.stability_score,
        "convergence_curve": {
            k: v.tolist() for k, v in results.convergence_curve.items()
        },
        "recommendations": results.recommendations,
    }