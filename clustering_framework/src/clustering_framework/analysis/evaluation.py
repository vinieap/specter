"""Evaluation module for clustering results.

This module provides comprehensive evaluation metrics and benchmarking tools
for assessing clustering quality and performance.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import tracemalloc

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)

@dataclass
class ClusteringMetrics:
    """Structured clustering evaluation metrics.
    
    Attributes:
        silhouette: Silhouette coefficient (-1 to 1, higher is better)
        calinski_harabasz: Calinski-Harabasz index (higher is better)
        davies_bouldin: Davies-Bouldin index (lower is better)
        inertia: Sum of squared distances to centers (if applicable)
    """
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    inertia: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Structured performance metrics.
    
    Attributes:
        fit_time: Time taken to fit the model (seconds)
        predict_time: Time taken for prediction (seconds)
        memory_usage: Peak memory usage (bytes)
        n_samples: Number of samples processed
        n_features: Number of features
    """
    fit_time: float
    predict_time: float
    memory_usage: int
    n_samples: int
    n_features: int

@dataclass
class ClusteringEvaluation:
    """Complete clustering evaluation results.
    
    Attributes:
        metrics: Clustering quality metrics
        performance: Performance metrics
        stability: Stability analysis results
        comparative: Comparative analysis with other algorithms
    """
    metrics: ClusteringMetrics
    performance: PerformanceMetrics
    stability: Dict[str, float]
    comparative: Optional[Dict[str, ClusteringMetrics]] = None

class ClusteringEvaluator:
    """Evaluates clustering algorithms for quality and performance."""
    
    def __init__(self, n_stability_runs: int = 5):
        """Initialize the evaluator.
        
        Args:
            n_stability_runs: Number of runs for stability analysis
        """
        self.n_stability_runs = n_stability_runs
        
    def evaluate(self, 
                 estimator: BaseEstimator,
                 X: np.ndarray,
                 comparison_estimators: Optional[Dict[str, BaseEstimator]] = None) -> ClusteringEvaluation:
        """Perform comprehensive evaluation of a clustering algorithm.
        
        Args:
            estimator: Clustering estimator to evaluate
            X: Input data array
            comparison_estimators: Optional dict of estimators to compare against
            
        Returns:
            ClusteringEvaluation object containing all evaluation results
        """
        # Measure performance
        performance = self._measure_performance(estimator, X)
        
        # Calculate clustering metrics
        labels = estimator.labels_
        metrics = self._calculate_metrics(X, labels, estimator)
        
        # Analyze stability
        stability = self._analyze_stability(estimator, X)
        
        # Compare with other algorithms
        comparative = None
        if comparison_estimators:
            comparative = {}
            for name, comp_estimator in comparison_estimators.items():
                comp_estimator.fit(X)
                comparative[name] = self._calculate_metrics(
                    X, comp_estimator.labels_, comp_estimator
                )
        
        return ClusteringEvaluation(
            metrics=metrics,
            performance=performance,
            stability=stability,
            comparative=comparative
        )
    
    def _measure_performance(self, estimator: BaseEstimator, X: np.ndarray) -> PerformanceMetrics:
        """Measure performance metrics of the clustering algorithm."""
        # Start memory tracking
        tracemalloc.start()
        
        # Measure fit time
        start_time = time.time()
        estimator.fit(X)
        fit_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        estimator.predict(X)
        predict_time = time.time() - start_time
        
        # Get memory usage
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            fit_time=fit_time,
            predict_time=predict_time,
            memory_usage=peak,
            n_samples=X.shape[0],
            n_features=X.shape[1]
        )
    
    def _calculate_metrics(self,
                          X: np.ndarray,
                          labels: np.ndarray,
                          estimator: BaseEstimator) -> ClusteringMetrics:
        """Calculate clustering quality metrics."""
        n_clusters = len(np.unique(labels))
        
        # Some metrics require at least 2 clusters
        if n_clusters < 2:
            return ClusteringMetrics(
                silhouette=0.0,
                calinski_harabasz=0.0,
                davies_bouldin=float('inf'),
                inertia=getattr(estimator, 'inertia_', None)
            )
        
        return ClusteringMetrics(
            silhouette=silhouette_score(X, labels),
            calinski_harabasz=calinski_harabasz_score(X, labels),
            davies_bouldin=davies_bouldin_score(X, labels),
            inertia=getattr(estimator, 'inertia_', None)
        )
    
    def _analyze_stability(self, estimator: BaseEstimator, X: np.ndarray) -> Dict[str, float]:
        """Analyze clustering stability through multiple runs."""
        base_labels = estimator.fit_predict(X)
        stability_scores = []
        ami_scores = []
        
        for _ in range(self.n_stability_runs):
            labels = estimator.fit_predict(X)
            stability_scores.append(adjusted_rand_score(base_labels, labels))
            ami_scores.append(adjusted_mutual_info_score(base_labels, labels))
            
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'mean_ami': np.mean(ami_scores),
            'std_ami': np.std(ami_scores)
        }