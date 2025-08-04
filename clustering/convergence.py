"""Convergence detection methods for optimization."""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class ConvergenceStatus:
    """Status of convergence detection."""
    converged: bool
    confidence: float  # 0 to 1
    method: str
    details: Dict[str, Any]
    recommendation: str


class ConvergenceDetector:
    """
    Advanced convergence detection for optimization processes.
    
    Features:
    - Multiple convergence detection methods
    - Statistical significance testing
    - Trend analysis
    - Plateau detection
    - Score stability analysis
    - Confidence estimation
    """

    def __init__(
        self,
        window_size: int = 10,
        min_evaluations: int = 20,
        significance_level: float = 0.05,
        improvement_threshold: float = 0.01,
    ):
        """Initialize convergence detector.

        Parameters
        ----------
        window_size : int, default=10
            Size of the sliding window for local convergence checks
        min_evaluations : int, default=20
            Minimum number of evaluations before checking convergence
        significance_level : float, default=0.05
            Statistical significance level for tests
        improvement_threshold : float, default=0.01
            Minimum improvement threshold for progress
        """
        self.window_size = window_size
        self.min_evaluations = min_evaluations
        self.significance_level = significance_level
        self.improvement_threshold = improvement_threshold

    def check_convergence(
        self,
        scores: List[float],
        times: List[float],
        n_total: int,
        best_score: float,
    ) -> ConvergenceStatus:
        """Check for convergence using multiple methods.

        Parameters
        ----------
        scores : List[float]
            History of optimization scores
        times : List[float]
            History of evaluation times
        n_total : int
            Total number of planned evaluations
        best_score : float
            Current best score

        Returns
        -------
        ConvergenceStatus
            Detailed convergence status and recommendations
        """
        if len(scores) < self.min_evaluations:
            return ConvergenceStatus(
                converged=False,
                confidence=0.0,
                method="insufficient_data",
                details={
                    "current_evals": len(scores),
                    "min_required": self.min_evaluations,
                },
                recommendation="Continue optimization to gather more data",
            )

        # Run all convergence checks
        methods = [
            self._check_score_stability,
            self._check_trend_convergence,
            self._check_plateau_detection,
            self._check_statistical_convergence,
            self._check_relative_improvement,
        ]

        # Collect results from all methods
        results = []
        for method in methods:
            result = method(scores, times, n_total, best_score)
            if result[0]:  # If method detected convergence
                results.append(result)

        if not results:
            return ConvergenceStatus(
                converged=False,
                confidence=0.0,
                method="no_convergence",
                details={
                    "recent_scores": scores[-self.window_size:],
                    "score_std": np.std(scores[-self.window_size:]),
                },
                recommendation="Continue optimization, no convergence detected",
            )

        # Select the most confident convergence result
        best_result = max(results, key=lambda x: x[1])
        converged, confidence, method, details, recommendation = best_result

        return ConvergenceStatus(
            converged=converged,
            confidence=confidence,
            method=method,
            details=details,
            recommendation=recommendation,
        )

    def _check_score_stability(
        self,
        scores: List[float],
        times: List[float],
        n_total: int,
        best_score: float,
    ) -> Tuple[bool, float, str, Dict[str, Any], str]:
        """Check for convergence based on score stability."""
        recent_scores = scores[-self.window_size:]
        score_std = np.std(recent_scores)
        score_range = np.max(recent_scores) - np.min(recent_scores)
        
        # Calculate stability metrics
        stability_ratio = score_std / (np.mean(recent_scores) + 1e-10)
        range_ratio = score_range / (best_score + 1e-10)
        
        # Define stability thresholds
        std_threshold = 0.01
        range_threshold = 0.05
        
        if stability_ratio < std_threshold and range_ratio < range_threshold:
            confidence = 1.0 - max(stability_ratio / std_threshold, range_ratio / range_threshold)
            return (
                True,
                confidence,
                "score_stability",
                {
                    "stability_ratio": stability_ratio,
                    "range_ratio": range_ratio,
                    "score_std": score_std,
                    "score_range": score_range,
                },
                "Scores have stabilized with low variance",
            )
        return (False, 0.0, "", {}, "")

    def _check_trend_convergence(
        self,
        scores: List[float],
        times: List[float],
        n_total: int,
        best_score: float,
    ) -> Tuple[bool, float, str, Dict[str, Any], str]:
        """Check for convergence based on score trend analysis."""
        if len(scores) < self.window_size * 2:
            return (False, 0.0, "", {}, "")

        # Fit linear regression to recent scores
        X = np.arange(self.window_size).reshape(-1, 1)
        y = scores[-self.window_size:]
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate trend metrics
        slope = model.coef_[0]
        slope_significance = abs(slope) / (np.std(y) + 1e-10)
        r2_score = model.score(X, y)
        
        # Check for flat or slightly positive trend
        if -0.001 <= slope <= 0.01 and r2_score > 0.7:
            confidence = r2_score * (1.0 - abs(slope))
            return (
                True,
                confidence,
                "trend_analysis",
                {
                    "slope": slope,
                    "r2_score": r2_score,
                    "slope_significance": slope_significance,
                },
                "Score trend has flattened with good fit",
            )
        return (False, 0.0, "", {}, "")

    def _check_plateau_detection(
        self,
        scores: List[float],
        times: List[float],
        n_total: int,
        best_score: float,
    ) -> Tuple[bool, float, str, Dict[str, Any], str]:
        """Check for convergence based on plateau detection."""
        if len(scores) < self.window_size * 3:
            return (False, 0.0, "", {}, "")

        # Compare statistics of consecutive windows
        window1 = scores[-2 * self.window_size:-self.window_size]
        window2 = scores[-self.window_size:]
        
        # Calculate window statistics
        mean_diff = abs(np.mean(window2) - np.mean(window1))
        std_diff = abs(np.std(window2) - np.std(window1))
        
        # Define plateau thresholds
        mean_threshold = 0.01
        std_threshold = 0.005
        
        if mean_diff < mean_threshold and std_diff < std_threshold:
            confidence = 1.0 - max(mean_diff / mean_threshold, std_diff / std_threshold)
            return (
                True,
                confidence,
                "plateau_detection",
                {
                    "mean_difference": mean_diff,
                    "std_difference": std_diff,
                    "window1_stats": {"mean": np.mean(window1), "std": np.std(window1)},
                    "window2_stats": {"mean": np.mean(window2), "std": np.std(window2)},
                },
                "Score has reached a stable plateau",
            )
        return (False, 0.0, "", {}, "")

    def _check_statistical_convergence(
        self,
        scores: List[float],
        times: List[float],
        n_total: int,
        best_score: float,
    ) -> Tuple[bool, float, str, Dict[str, Any], str]:
        """Check for convergence using statistical tests."""
        if len(scores) < self.window_size * 2:
            return (False, 0.0, "", {}, "")

        # Perform statistical tests on consecutive windows
        window1 = scores[-2 * self.window_size:-self.window_size]
        window2 = scores[-self.window_size:]
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_statistic, ks_pvalue = stats.ks_2samp(window1, window2)
        
        # Mann-Whitney U test for median comparison
        mw_statistic, mw_pvalue = stats.mannwhitneyu(
            window1, window2, alternative='two-sided'
        )
        
        # Check if distributions are similar
        if ks_pvalue > self.significance_level and mw_pvalue > self.significance_level:
            confidence = min(ks_pvalue, mw_pvalue)
            return (
                True,
                confidence,
                "statistical_tests",
                {
                    "ks_test": {"statistic": ks_statistic, "pvalue": ks_pvalue},
                    "mw_test": {"statistic": mw_statistic, "pvalue": mw_pvalue},
                },
                "Score distributions are statistically similar",
            )
        return (False, 0.0, "", {}, "")

    def _check_relative_improvement(
        self,
        scores: List[float],
        times: List[float],
        n_total: int,
        best_score: float,
    ) -> Tuple[bool, float, str, Dict[str, Any], str]:
        """Check for convergence based on relative improvement rate."""
        if len(scores) < self.window_size * 2:
            return (False, 0.0, "", {}, "")

        # Calculate relative improvements
        improvements = np.diff(scores[-self.window_size:])
        relative_improvements = improvements / (best_score + 1e-10)
        
        # Calculate improvement statistics
        mean_improvement = np.mean(relative_improvements)
        max_improvement = np.max(relative_improvements)
        
        if max_improvement < self.improvement_threshold:
            confidence = 1.0 - (max_improvement / self.improvement_threshold)
            return (
                True,
                confidence,
                "relative_improvement",
                {
                    "mean_improvement": mean_improvement,
                    "max_improvement": max_improvement,
                    "recent_improvements": relative_improvements.tolist(),
                },
                "No significant improvements in recent evaluations",
            )
        return (False, 0.0, "", {}, "")