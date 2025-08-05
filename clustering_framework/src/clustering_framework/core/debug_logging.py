"""
Debug logging utilities for the clustering framework.

This module provides comprehensive debug logging capabilities with function tracing,
performance monitoring, and detailed state tracking.
"""

import functools
import inspect
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Generator, Optional, TypeVar, cast

import numpy as np

from .logging import get_logger

T = TypeVar("T")


@dataclass
class DebugInfo:
    """Container for debug information."""

    function_name: str
    module_name: str
    args: tuple
    kwargs: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    return_value: Any = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    call_stack: Optional[str] = None


class DebugLogger:
    """
    Advanced debug logging with function tracing and performance monitoring.
    """

    def __init__(self, name: str = "debug") -> None:
        """
        Initialize the debug logger.

        Args:
            name: Logger name prefix
        """
        self.logger = get_logger(f"{name}_debug")
        self.debug_enabled = False
        self.trace_enabled = False
        self.performance_enabled = False
        self._call_depth = 0

    def enable_debug(self) -> None:
        """Enable debug logging."""
        self.debug_enabled = True

    def disable_debug(self) -> None:
        """Disable debug logging."""
        self.debug_enabled = False

    def enable_trace(self) -> None:
        """Enable function call tracing."""
        self.trace_enabled = True

    def disable_trace(self) -> None:
        """Disable function call tracing."""
        self.trace_enabled = False

    def enable_performance(self) -> None:
        """Enable performance monitoring."""
        self.performance_enabled = True

    def disable_performance(self) -> None:
        """Disable performance monitoring."""
        self.performance_enabled = False

    def _format_value(self, value: Any) -> str:
        """Format a value for debug output."""
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        return str(value)

    def _format_args(self, args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Format function arguments for debug output."""
        formatted_args = {
            f"arg_{i}": self._format_value(arg) for i, arg in enumerate(args)
        }
        formatted_kwargs = {k: self._format_value(v) for k, v in kwargs.items()}
        return {**formatted_args, **formatted_kwargs}

    def debug_function(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for comprehensive function debugging.

        Args:
            func: Function to debug

        Returns:
            Decorated function
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.debug_enabled:
                return func(*args, **kwargs)

            debug_info = DebugInfo(
                function_name=func.__name__,
                module_name=func.__module__ or "",
                args=args,
                kwargs=kwargs,
                start_time=time.time(),
                call_stack=str(inspect.stack()[1:]) if self.trace_enabled else None,
            )

            indent = "  " * self._call_depth
            self._call_depth += 1

            try:
                if self.trace_enabled:
                    self.logger.debug(
                        f"{indent}Entering {func.__name__}",
                        extra={
                            "args": self._format_args(args, kwargs),
                            "call_stack": debug_info.call_stack,
                        },
                    )

                result = func(*args, **kwargs)
                debug_info.return_value = result

                return result

            except Exception:
                self.logger.exception(
                    f"{indent}Error in {func.__name__}",
                    extra={"debug_info": asdict(debug_info)},
                )
                raise

            finally:
                self._call_depth -= 1
                debug_info.end_time = time.time()
                debug_info.execution_time = debug_info.end_time - debug_info.start_time

                if self.performance_enabled:
                    self.logger.debug(
                        f"{indent}Performance stats for {func.__name__}",
                        extra={
                            "execution_time": debug_info.execution_time,
                            "memory_usage": debug_info.memory_usage,
                        },
                    )

                if self.trace_enabled:
                    self.logger.debug(
                        f"{indent}Exiting {func.__name__}",
                        extra={
                            "execution_time": debug_info.execution_time,
                            "return_value": self._format_value(debug_info.return_value),
                        },
                    )

        return cast(Callable[..., T], wrapper)

    @contextmanager
    def debug_context(
        self, name: str, extra: Optional[Dict[str, Any]] = None
    ) -> Generator[None, None, None]:
        """
        Context manager for debugging a block of code.

        Args:
            name: Name for the debug context
            extra: Additional debug information

        Example:
            >>> with debug_logger.debug_context("data_processing"):
            ...     process_data()
        """
        if not self.debug_enabled:
            yield
            return

        start_time = time.time()
        indent = "  " * self._call_depth
        self._call_depth += 1

        try:
            self.logger.debug(f"{indent}Entering context: {name}", extra=extra)
            yield

        except Exception:
            self.logger.exception(f"{indent}Error in context: {name}", extra=extra)
            raise

        finally:
            self._call_depth -= 1
            execution_time = time.time() - start_time

            if self.performance_enabled:
                self.logger.debug(
                    f"{indent}Performance stats for context: {name}",
                    extra={
                        "execution_time": execution_time,
                        **(extra if extra is not None else {}),
                    },
                )

            self.logger.debug(
                f"{indent}Exiting context: {name}",
                extra={
                    "execution_time": execution_time,
                    **(extra if extra is not None else {}),
                },
            )


# Global debug logger instance
_default_debug_logger: Optional[DebugLogger] = None


def get_debug_logger(name: str = "debug") -> DebugLogger:
    """
    Get or create the global debug logger instance.

    Args:
        name: Logger name prefix

    Returns:
        The global DebugLogger instance
    """
    global _default_debug_logger

    if _default_debug_logger is None:
        _default_debug_logger = DebugLogger(name)

    return _default_debug_logger
