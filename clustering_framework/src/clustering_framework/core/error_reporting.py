"""
Error reporting system for the clustering framework.

This module provides functionality for collecting, aggregating, and reporting errors
that occur during the execution of clustering algorithms and optimization processes.
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .exceptions import ClusteringError
from .logging import get_logger


@dataclass
class ErrorReport:
    """Container for error report information."""

    error_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    count: int = 1
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    related_errors: List[Dict[str, Any]] = field(default_factory=list)


class ErrorReporter:
    """
    Manages error reporting and aggregation for the clustering framework.
    """

    def __init__(
        self,
        report_dir: Optional[Path] = None,
        max_reports: int = 1000,
        aggregate_similar: bool = True,
    ) -> None:
        """
        Initialize the error reporter.

        Args:
            report_dir: Directory to store error reports
            max_reports: Maximum number of reports to keep in memory
            aggregate_similar: Whether to aggregate similar errors
        """
        self.logger = get_logger()
        self.report_dir = Path(report_dir) if report_dir else None
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)

        self.max_reports = max_reports
        self.aggregate_similar = aggregate_similar
        self.reports: Dict[str, ErrorReport] = {}
        self.error_patterns: Dict[str, Set[str]] = defaultdict(set)

    def _get_error_key(self, error: Exception) -> str:
        """Generate a unique key for an error."""
        if isinstance(error, ClusteringError):
            return f"{error.__class__.__name__}:{str(error)}"
        return f"{error.__class__.__name__}:{str(error)}"

    def _are_errors_similar(self, error1: ErrorReport, error2: ErrorReport) -> bool:
        """
        Check if two errors are similar enough to be aggregated.

        This is a simple implementation that can be extended with more
        sophisticated similarity metrics if needed.
        """
        return (
            error1.error_type == error2.error_type and error1.message == error2.message
        )

    def report_error(
        self, error: Exception, additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Report an error for tracking and analysis.

        Args:
            error: The exception to report
            additional_context: Additional context to include in the report
        """
        error_key = self._get_error_key(error)
        context = {}

        if isinstance(error, ClusteringError):
            context.update(error.context)

        if additional_context:
            context.update(additional_context)

        now = datetime.now()

        if error_key in self.reports:
            report = self.reports[error_key]
            report.count += 1
            report.last_occurrence = now
            if additional_context:
                report.related_errors.append(
                    {"timestamp": now.isoformat(), "context": additional_context}
                )
        else:
            report = ErrorReport(
                error_type=error.__class__.__name__,
                message=str(error),
                context=context,
                first_occurrence=now,
                last_occurrence=now,
            )

            if len(self.reports) >= self.max_reports:
                # Remove oldest report
                oldest_key = min(
                    self.reports.keys(), key=lambda k: self.reports[k].last_occurrence
                )
                del self.reports[oldest_key]

            self.reports[error_key] = report

            if self.aggregate_similar:
                # Check for similar existing errors
                for existing_key, existing_report in self.reports.items():
                    if existing_key != error_key and self._are_errors_similar(
                        existing_report, report
                    ):
                        self.error_patterns[existing_key].add(error_key)

        # Log the error
        self.logger.error(
            f"Error reported: {error_key}", extra={"error_report": asdict(report)}
        )

        # Save report to file if directory is configured
        if self.report_dir:
            self._save_report(error_key, report)

    def _save_report(self, error_key: str, report: ErrorReport) -> None:
        """Save an error report to a file."""
        report_file = self.report_dir / f"{error_key.replace(':', '_')}.json"
        report_data = asdict(report)
        report_data["timestamp"] = report_data["timestamp"].isoformat()
        report_data["first_occurrence"] = report_data["first_occurrence"].isoformat()
        report_data["last_occurrence"] = report_data["last_occurrence"].isoformat()

        with report_file.open("w") as f:
            json.dump(report_data, f, indent=2)

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all reported errors.

        Returns:
            Dictionary containing error statistics and patterns
        """
        error_types = defaultdict(int)
        total_errors = 0
        error_patterns = {}

        for error_key, report in self.reports.items():
            error_types[report.error_type] += report.count
            total_errors += report.count

            if error_key in self.error_patterns:
                error_patterns[error_key] = list(self.error_patterns[error_key])

        return {
            "total_errors": total_errors,
            "error_types": dict(error_types),
            "error_patterns": error_patterns,
            "unique_errors": len(self.reports),
        }

    def get_most_frequent_errors(
        self, limit: int = 10
    ) -> List[Tuple[str, ErrorReport]]:
        """
        Get the most frequently occurring errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of (error_key, report) tuples, sorted by frequency
        """
        return sorted(self.reports.items(), key=lambda x: x[1].count, reverse=True)[
            :limit
        ]

    def clear_reports(self) -> None:
        """Clear all stored error reports."""
        self.reports.clear()
        self.error_patterns.clear()


# Global error reporter instance
_default_reporter: Optional[ErrorReporter] = None


def get_error_reporter(**kwargs: Any) -> ErrorReporter:
    """
    Get or create the global error reporter instance.

    Args:
        **kwargs: Arguments passed to ErrorReporter constructor

    Returns:
        The global ErrorReporter instance
    """
    global _default_reporter

    if _default_reporter is None:
        _default_reporter = ErrorReporter(**kwargs)

    return _default_reporter
