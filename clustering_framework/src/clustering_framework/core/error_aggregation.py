"""
Error aggregation system for the clustering framework.

This module provides functionality for aggregating and analyzing errors across
multiple runs, processes, and components of the clustering framework.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .error_reporting import get_error_reporter
from .exceptions import (
    AlgorithmError,
    ConfigurationError,
    DataError,
    MetricError,
    ParallelizationError,
    PersistenceError,
    ResourceError,
    ValidationError,
)
from .logging import get_logger


class ErrorSeverity(Enum):
    """Severity levels for aggregated errors."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ErrorCategory(Enum):
    """Categories for grouping similar errors."""

    CONFIGURATION = auto()
    VALIDATION = auto()
    ALGORITHM = auto()
    OPTIMIZATION = auto()
    RESOURCE = auto()
    DATA = auto()
    METRIC = auto()
    PARALLELIZATION = auto()
    PERSISTENCE = auto()
    EXTERNAL = auto()
    OTHER = auto()


@dataclass
class ErrorPattern:
    """Pattern information for similar errors."""

    pattern_id: str
    error_type: str
    message_pattern: str
    frequency: int = 0
    last_occurrence: datetime = field(default_factory=datetime.now)
    affected_components: Set[str] = field(default_factory=set)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.OTHER
    related_errors: List[str] = field(default_factory=list)


class ErrorAggregator:
    """
    Aggregates and analyzes errors across the clustering framework.
    """

    def __init__(
        self, time_window: timedelta = timedelta(hours=1), pattern_threshold: int = 3
    ) -> None:
        """
        Initialize the error aggregator.

        Args:
            time_window: Time window for error aggregation
            pattern_threshold: Minimum occurrences to identify a pattern
        """
        self.logger = get_logger()
        self.error_reporter = get_error_reporter()
        self.time_window = time_window
        self.pattern_threshold = pattern_threshold
        self.patterns: Dict[str, ErrorPattern] = {}
        self.component_errors: Dict[str, List[str]] = defaultdict(list)
        self.error_timeline: List[Tuple[datetime, str]] = []
        self._error_categories: Dict[type, ErrorCategory] = self._init_categories()

    def _init_categories(self) -> Dict[type, ErrorCategory]:
        """Initialize error type to category mapping."""
        from . import exceptions

        return {
            exceptions.ConfigurationError: ErrorCategory.CONFIGURATION,
            exceptions.ValidationError: ErrorCategory.VALIDATION,
            exceptions.AlgorithmError: ErrorCategory.ALGORITHM,
            exceptions.OptimizationError: ErrorCategory.OPTIMIZATION,
            exceptions.ResourceError: ErrorCategory.RESOURCE,
            exceptions.DataError: ErrorCategory.DATA,
            exceptions.MetricError: ErrorCategory.METRIC,
            exceptions.ParallelizationError: ErrorCategory.PARALLELIZATION,
            exceptions.PersistenceError: ErrorCategory.PERSISTENCE,
            exceptions.ExternalError: ErrorCategory.EXTERNAL,
        }

    def _get_error_category(self, error: Exception) -> ErrorCategory:
        """Get the category for an error type."""
        for error_type, category in self._error_categories.items():
            if isinstance(error, error_type):
                return category
        return ErrorCategory.OTHER

    def _get_error_severity(self, error: Exception, frequency: int) -> ErrorSeverity:
        """
        Determine error severity based on type and frequency.

        Args:
            error: The error to analyze
            frequency: How often the error has occurred

        Returns:
            Appropriate error severity level
        """
        # Critical errors
        if any(
            [
                isinstance(error, ResourceError),  # Resource exhaustion
                isinstance(error, PersistenceError),  # Data loss risk
                isinstance(error, ParallelizationError),  # System stability
                frequency >= 10,  # High frequency indicates serious issue
            ]
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if any(
            [
                isinstance(error, AlgorithmError),  # Core functionality
                isinstance(error, DataError),  # Data integrity
                isinstance(error, ValidationError),  # Invalid state
                frequency >= 5,  # Moderate frequency
            ]
        ):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if any(
            [
                isinstance(error, ConfigurationError),  # Config issues
                isinstance(error, MetricError),  # Metric computation
                frequency >= 3,  # Low frequency
            ]
        ):
            return ErrorSeverity.MEDIUM

        # Low severity errors
        return ErrorSeverity.LOW

    def _create_pattern_id(self, error: Exception) -> str:
        """Create a unique pattern ID for an error."""
        return f"{error.__class__.__name__}_{hash(str(error))}"

    def aggregate_error(
        self, error: Exception, component: str, context: Optional[Dict[str, Any]] = None
    ) -> ErrorPattern:
        """
        Aggregate an error into patterns.

        Args:
            error: The error to aggregate
            component: Component where the error occurred
            context: Additional context information

        Returns:
            The error pattern for this error
        """
        pattern_id = self._create_pattern_id(error)
        now = datetime.now()

        # Clean up old entries
        cutoff = now - self.time_window
        self.error_timeline = [(t, e) for t, e in self.error_timeline if t > cutoff]

        # Add new error
        self.error_timeline.append((now, pattern_id))
        self.component_errors[component].append(pattern_id)

        # Update or create pattern
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_occurrence = now
            pattern.affected_components.add(component)
        else:
            pattern = ErrorPattern(
                pattern_id=pattern_id,
                error_type=error.__class__.__name__,
                message_pattern=str(error),
                frequency=1,
                last_occurrence=now,
                affected_components={component},
                category=self._get_error_category(error),
            )
            self.patterns[pattern_id] = pattern

        # Update severity based on frequency
        pattern.severity = self._get_error_severity(error, pattern.frequency)

        # Log pattern update
        self.logger.info(
            f"Error pattern updated: {pattern_id}",
            extra={
                "pattern": {
                    "id": pattern_id,
                    "frequency": pattern.frequency,
                    "severity": pattern.severity.name,
                    "category": pattern.category.name,
                    "components": list(pattern.affected_components),
                }
            },
        )

        return pattern

    def get_active_patterns(
        self,
        min_frequency: Optional[int] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
    ) -> List[ErrorPattern]:
        """
        Get active error patterns matching criteria.

        Args:
            min_frequency: Minimum occurrence frequency
            severity: Filter by severity level
            category: Filter by error category

        Returns:
            List of matching error patterns
        """
        min_frequency = min_frequency or self.pattern_threshold
        cutoff = datetime.now() - self.time_window

        return [
            pattern
            for pattern in self.patterns.values()
            if (
                pattern.frequency >= min_frequency
                and pattern.last_occurrence > cutoff
                and (severity is None or pattern.severity == severity)
                and (category is None or pattern.category == category)
            )
        ]

    def get_component_health(
        self, component: str, window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get health metrics for a component.

        Args:
            component: Component to analyze
            window: Custom time window (uses instance window if None)

        Returns:
            Dictionary with health metrics
        """
        window = window or self.time_window
        cutoff = datetime.now() - window

        # Get recent errors for component
        recent_errors = [
            pattern_id
            for pattern_id in self.component_errors[component]
            if pattern_id in self.patterns
            and self.patterns[pattern_id].last_occurrence > cutoff
        ]

        # Count errors by severity
        severity_counts = defaultdict(int)
        for pattern_id in recent_errors:
            pattern = self.patterns[pattern_id]
            severity_counts[pattern.severity.name] += pattern.frequency

        return {
            "component": component,
            "total_errors": len(recent_errors),
            "unique_patterns": len(set(recent_errors)),
            "severity_distribution": dict(severity_counts),
            "has_critical": severity_counts[ErrorSeverity.CRITICAL.name] > 0,
            "error_rate": len(recent_errors) / window.total_seconds(),
        }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.

        Returns:
            Dictionary with system health metrics
        """
        cutoff = datetime.now() - self.time_window
        active_patterns = self.get_active_patterns()

        # Aggregate metrics
        total_errors = sum(p.frequency for p in active_patterns)
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        component_counts = defaultdict(int)

        for pattern in active_patterns:
            severity_counts[pattern.severity.name] += pattern.frequency
            category_counts[pattern.category.name] += pattern.frequency
            for component in pattern.affected_components:
                component_counts[component] += pattern.frequency

        return {
            "total_errors": total_errors,
            "unique_patterns": len(active_patterns),
            "severity_distribution": dict(severity_counts),
            "category_distribution": dict(category_counts),
            "component_distribution": dict(component_counts),
            "has_critical": severity_counts[ErrorSeverity.CRITICAL.name] > 0,
            "error_rate": total_errors / self.time_window.total_seconds(),
        }


# Global error aggregator instance
_default_aggregator: Optional[ErrorAggregator] = None


def get_error_aggregator(**kwargs: Any) -> ErrorAggregator:
    """
    Get or create the global error aggregator instance.

    Args:
        **kwargs: Arguments passed to ErrorAggregator constructor

    Returns:
        The global ErrorAggregator instance
    """
    global _default_aggregator

    if _default_aggregator is None:
        _default_aggregator = ErrorAggregator(**kwargs)

    return _default_aggregator
