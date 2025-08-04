"""Core type definitions for the clustering library."""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class OptimizationResult:
    """Results from a clustering optimization run.
    
    This class provides a structured way to access optimization results,
    including the best model found, its parameters, and various metrics
    about the optimization process.
    
    Attributes
    ----------
    best_score : float
        The score of the best model found
    best_params : Dict[str, Any]
        Parameters that achieved the best score
    best_model : BaseEstimator
        The best clustering model found
    history : List[Dict[str, Any]]
        History of all evaluations performed
    convergence_info : Dict[str, Any]
        Information about optimization convergence
    execution_stats : Dict[str, Any]
        Statistics about the optimization run
    """
    best_score: float
    best_params: Dict[str, Any]
    best_model: BaseEstimator
    history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_stats: Dict[str, Any]


@dataclass
class NoiseAnalysis:
    """Results from noise detection analysis.
    
    This class provides structured access to noise analysis results,
    including detected noise points, their types, and recommendations
    for handling the noise.
    
    Attributes
    ----------
    noise_ratio : float
        Ratio of noise points to total points (0 to 1)
    noise_indices : np.ndarray
        Indices of points identified as noise
    noise_types : Dict[str, np.ndarray]
        Classification of noise points by type
    recommendations : Dict[str, List[str]]
        Recommendations for handling detected noise
    details : Dict[str, Any]
        Detailed analysis results and metrics
    """
    noise_ratio: float
    noise_indices: np.ndarray
    noise_types: Dict[str, np.ndarray]
    recommendations: Dict[str, List[str]]
    details: Dict[str, Any]


@dataclass
class ConvergenceStatus:
    """Status of optimization convergence detection.
    
    This class provides information about whether optimization has converged,
    how confident we are in the convergence, and recommendations for
    continuing or stopping optimization.
    
    Attributes
    ----------
    converged : bool
        Whether optimization has converged
    confidence : float
        Confidence in convergence detection (0 to 1)
    method : str
        Method used to detect convergence
    details : Dict[str, Any]
        Detailed convergence metrics
    recommendation : str
        Recommendation for optimization process
    """
    converged: bool
    confidence: float
    method: str
    details: Dict[str, Any]
    recommendation: str


@dataclass
class AlgorithmPerformance:
    """Performance metrics for a clustering algorithm.
    
    This class aggregates performance metrics for a clustering algorithm
    across multiple optimization runs or validation sets.
    
    Attributes
    ----------
    algorithm : str
        Name of the algorithm
    mean_score : float
        Average performance score
    std_score : float
        Standard deviation of scores
    best_score : float
        Best score achieved
    worst_score : float
        Worst score achieved
    mean_time : float
        Average execution time
    mean_evaluations : float
        Average number of evaluations needed
    best_params : Dict[str, Any]
        Parameters that achieved the best score
    success_rate : float
        Rate of successful clustering attempts
    parameter_importance : Dict[str, float]
        Importance scores for each parameter
    convergence_rate : float
        Rate of convergence across runs
    score_stability : float
        Measure of score stability
    """
    algorithm: str
    mean_score: float
    std_score: float
    best_score: float
    worst_score: float
    mean_time: float
    mean_evaluations: float
    best_params: Dict[str, Any]
    success_rate: float
    parameter_importance: Dict[str, float]
    convergence_rate: float
    score_stability: float


# Type aliases for common types
ParamDict = Dict[str, Any]
OptionalParamDict = Optional[ParamDict]
ArrayLike = Union[np.ndarray, List[float], List[List[float]]]