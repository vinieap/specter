"""Convergence detection functionality."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import silhouette_score


class ConvergenceCriterion(Enum):
    """Available convergence criteria."""

    INERTIA = auto()  # For algorithms with inertia (e.g., K-means)
    SILHOUETTE = auto()  # Silhouette score
    STABILITY = auto()  # Label stability between iterations
    CUSTOM = auto()  # Custom metric


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection.

    Attributes:
        criteria: List of convergence criteria to monitor
        thresholds: Thresholds for each criterion
        patience: Number of iterations without improvement before early stopping
        min_iterations: Minimum number of iterations before convergence
        max_iterations: Maximum number of iterations
        window_size: Window size for moving average
        custom_metric: Optional custom metric function
    """

    criteria: List[ConvergenceCriterion]
    thresholds: Dict[ConvergenceCriterion, float]
    patience: int = 5
    min_iterations: int = 2
    max_iterations: int = 100
    window_size: int = 3
    custom_metric: Optional[Callable[[BaseEstimator, np.ndarray], float]] = None

    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.criteria, list):
            raise ValueError("criteria must be a list")
        if not self.criteria:
            raise ValueError("At least one convergence criterion must be specified")

        # Validate criteria types
        for criterion in self.criteria:
            if not isinstance(criterion, ConvergenceCriterion):
                raise ValueError(f"Invalid criterion type: {criterion}")

        # Check thresholds
        for criterion in self.criteria:
            if criterion not in self.thresholds:
                raise ValueError(f"Threshold not specified for criterion: {criterion}")

        # Validate numeric parameters
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.min_iterations < 1:
            raise ValueError("min_iterations must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")

        # Validate parameter relationships
        if self.min_iterations >= self.max_iterations:
            raise ValueError("min_iterations must be less than max_iterations")

        # Check custom metric
        if ConvergenceCriterion.CUSTOM in self.criteria:
            if self.custom_metric is None:
                raise ValueError("custom_metric must be provided for CUSTOM criterion")


@dataclass
class ConvergenceState:
    """State for convergence detection.

    Attributes:
        iteration: Current iteration number
        converged: Whether convergence has been reached
        convergence_reason: Reason for convergence
        metrics: Current metric values
        best_metrics: Best metric values seen
        history: History of metric values
        iterations_without_improvement: Number of iterations without improvement
    """

    iteration: int = 0
    converged: bool = False
    convergence_reason: Optional[str] = None
    metrics: Dict[ConvergenceCriterion, float] = field(default_factory=dict)
    best_metrics: Dict[ConvergenceCriterion, float] = field(default_factory=dict)
    history: Dict[ConvergenceCriterion, List[float]] = field(default_factory=dict)
    iterations_without_improvement: int = 0


class ConvergenceDetector:
    """Convergence detection for clustering algorithms.

    This class handles convergence detection using multiple criteria and
    provides early stopping capabilities.
    """

    def __init__(self, config: ConvergenceConfig):
        """Initialize convergence detector.

        Args:
            config: Convergence detection configuration
        """
        self.config = config

    def initialize(self) -> ConvergenceState:
        """Initialize convergence state.

        Returns:
            Initial convergence state
        """
        state = ConvergenceState()

        # Initialize metrics
        for criterion in self.config.criteria:
            state.metrics[criterion] = float('inf')
            state.best_metrics[criterion] = float('inf')
            state.history[criterion] = []

        return state

    def _compute_metric(
        self,
        criterion: ConvergenceCriterion,
        estimator: BaseEstimator,
        X: np.ndarray,
    ) -> float:
        """Compute metric value for a criterion.

        Args:
            criterion: Convergence criterion
            estimator: Fitted estimator
            X: Input data matrix

        Returns:
            Metric value
        """
        if criterion == ConvergenceCriterion.INERTIA:
            if hasattr(estimator, "inertia_"):
                return estimator.inertia_
            else:
                raise ValueError("Estimator does not have inertia_")

        elif criterion == ConvergenceCriterion.SILHOUETTE:
            labels = (
                estimator.predict(X)
                if hasattr(estimator, "predict")
                else estimator.labels_
            )
            return -silhouette_score(X, labels)  # Negative for minimization

        elif criterion == ConvergenceCriterion.STABILITY:
            if not hasattr(estimator, "labels_"):
                raise ValueError("Estimator does not have labels_")
            return np.mean(estimator.labels_ == self._previous_labels)

        elif criterion == ConvergenceCriterion.CUSTOM:
            if self.config.custom_metric is None:
                raise ValueError("custom_metric not provided")
            return self.config.custom_metric(estimator, X)

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def update(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        state: ConvergenceState,
    ) -> ConvergenceState:
        """Update convergence state.

        Args:
            estimator: Fitted estimator
            X: Input data matrix
            state: Current convergence state

        Returns:
            Updated convergence state
        """
        state.iteration += 1

        # Store labels for stability computation
        if ConvergenceCriterion.STABILITY in self.config.criteria:
            self._previous_labels = (
                estimator.predict(X)
                if hasattr(estimator, "predict")
                else estimator.labels_
            )

        # Compute metrics
        improved = False
        for criterion in self.config.criteria:
            metric_value = self._compute_metric(criterion, estimator, X)
            state.metrics[criterion] = metric_value
            state.history[criterion].append(metric_value)

            # Check for improvement
            if metric_value < state.best_metrics[criterion]:
                state.best_metrics[criterion] = metric_value
                improved = True

        # Update improvement counter
        if improved:
            state.iterations_without_improvement = 0
        else:
            state.iterations_without_improvement += 1

        # Check convergence conditions
        if state.iteration >= self.config.min_iterations:
            # Check early stopping
            if (
                state.iterations_without_improvement >= self.config.patience
                and state.iteration >= self.config.min_iterations
            ):
                state.converged = True
                state.convergence_reason = "No improvement for {} iterations".format(
                    self.config.patience
                )
                return state

            # Check thresholds
            all_converged = True
            for criterion in self.config.criteria:
                threshold = self.config.thresholds[criterion]
                history = state.history[criterion][-self.config.window_size:]

                if len(history) >= self.config.window_size:
                    mean_value = np.mean(history)
                    if mean_value > threshold:
                        all_converged = False
                        break
                else:
                    all_converged = False
                    break

            if all_converged:
                state.converged = True
                state.convergence_reason = "All criteria met thresholds"

        # Check max iterations
        if state.iteration >= self.config.max_iterations:
            state.converged = True
            state.convergence_reason = "Maximum iterations reached"

        return state