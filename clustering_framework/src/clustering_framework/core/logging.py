"""
Structured logging system for the clustering framework.

This module provides a comprehensive logging system with structured output,
rotation support, and various log handlers for different use cases.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .error_context import get_full_context
from .logging_config import LogFormat, LogHandlerConfig, get_logging_config


class StructuredLogger:
    """
    A structured logger that outputs JSON-formatted logs with context information.
    """

    def __init__(self, name: str, config: Optional[LogHandlerConfig] = None) -> None:
        """
        Initialize the structured logger.

        Args:
            name: Logger name
            config: Logger configuration
        """
        self.logger = logging.getLogger(name)
        self.config = config or get_logging_config().get_component_config(name)

        # Configure logger
        self.logger.setLevel(self.config.level.value)
        self.logger.propagate = self.config.propagate

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add configured handlers
        for handler_config in self.config.handlers:
            if not handler_config or not handler_config.enabled:
                continue

            if handler_config.filename:
                # File handler
                log_dir = get_logging_config().log_dir
                if log_dir:
                    log_dir = Path(log_dir)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_file = log_dir / handler_config.filename
                    handler = logging.handlers.RotatingFileHandler(
                        log_file,
                        maxBytes=handler_config.max_bytes,
                        backupCount=handler_config.backup_count,
                    )
            else:
                # Console handler
                handler = logging.StreamHandler(sys.stdout)
                if handler_config.use_colors:
                    self._add_colors(handler)

            # Set handler level and formatter
            handler.setLevel(handler_config.level.value)
            if handler_config.format == LogFormat.JSON:
                formatter = self._create_json_formatter()
            else:
                formatter = self._create_text_formatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _create_json_formatter(self) -> logging.Formatter:
        """Create a JSON formatter."""

        def format_func(record: logging.LogRecord) -> str:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "context": get_full_context(),
            }

            if hasattr(record, "extra"):
                log_entry["extra"] = record.extra

            # Add configured extra fields
            log_entry.update(self.config.extra_fields)

            if record.exc_info:
                log_entry["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": self._format_traceback(record.exc_info[2]),
                }

            return json.dumps(log_entry)

        return logging.Formatter(fmt=format_func)

    def _create_text_formatter(self) -> logging.Formatter:
        """Create a text formatter."""
        return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def _add_colors(self, handler: logging.Handler) -> None:
        """Add colors to console output."""
        if hasattr(handler, "setFormatter"):
            colors = {
                "DEBUG": "\033[36m",  # Cyan
                "INFO": "\033[32m",  # Green
                "WARNING": "\033[33m",  # Yellow
                "ERROR": "\033[31m",  # Red
                "CRITICAL": "\033[35m",  # Magenta
                "RESET": "\033[0m",  # Reset
            }

            original_factory = handler.formatter._style._fmt
            handler.formatter._style._fmt = (
                f"%(asctime)s - %(name)s - {colors['%(levelname)s']}%(levelname)s"
                f"{colors['RESET']} - %(message)s"
            )

    def _format_traceback(self, tb: Any) -> str:
        """Format a traceback for JSON output."""
        import traceback

        return "".join(traceback.format_tb(tb))

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        self.logger.debug(message, extra={"extra": extra} if extra else None)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message."""
        self.logger.info(message, extra={"extra": extra} if extra else None)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self.logger.warning(message, extra={"extra": extra} if extra else None)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        self.logger.error(message, extra={"extra": extra} if extra else None)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a critical message."""
        self.logger.critical(message, extra={"extra": extra} if extra else None)

    def exception(
        self,
        message: str,
        exc_info: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an exception with traceback."""
        self.logger.exception(
            message, exc_info=exc_info, extra={"extra": extra} if extra else None
        )


# Global logger instance
_default_logger: Optional[StructuredLogger] = None


def get_logger(
    name: Optional[str] = None, config: Optional[LogHandlerConfig] = None
) -> StructuredLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (defaults to 'clustering_framework')
        config: Logger configuration

    Returns:
        A StructuredLogger instance
    """
    global _default_logger

    if name is None:
        name = "clustering_framework"

    if _default_logger is None or name != _default_logger.logger.name:
        _default_logger = StructuredLogger(name, config)

    return _default_logger
