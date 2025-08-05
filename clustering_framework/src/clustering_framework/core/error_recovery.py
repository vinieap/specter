"""
Error recovery mechanisms for the clustering framework.

This module provides utilities for handling and recovering from errors during
clustering and optimization processes.
"""

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, cast

from .error_context import error_context
from .error_reporting import get_error_reporter
from .logging import get_logger

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""

    RETRY = auto()  # Retry the operation
    FALLBACK = auto()  # Use fallback value/operation
    SKIP = auto()  # Skip the operation
    ABORT = auto()  # Abort the process


@dataclass
class RecoveryAction:
    """Defines an error recovery action."""

    strategy: RecoveryStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_value: Any = None
    cleanup_func: Optional[Callable[[], None]] = None


class RecoveryManager:
    """
    Manages error recovery strategies and actions.
    """

    def __init__(self) -> None:
        """Initialize the recovery manager."""
        self.logger = get_logger()
        self.error_reporter = get_error_reporter()
        self._recovery_handlers: Dict[type, List[RecoveryAction]] = {}

    def register_handler(
        self,
        error_type: type,
        strategy: RecoveryStrategy,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_value: Any = None,
        cleanup_func: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Register a recovery handler for an error type.

        Args:
            error_type: Type of error to handle
            strategy: Recovery strategy to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            fallback_value: Value to use for FALLBACK strategy
            cleanup_func: Function to call for cleanup before recovery
        """
        action = RecoveryAction(
            strategy=strategy,
            max_retries=max_retries,
            retry_delay=retry_delay,
            fallback_value=fallback_value,
            cleanup_func=cleanup_func,
        )

        if error_type not in self._recovery_handlers:
            self._recovery_handlers[error_type] = []

        self._recovery_handlers[error_type].append(action)

    def get_handlers(self, error: Exception) -> List[RecoveryAction]:
        """
        Get all applicable recovery handlers for an error.

        Args:
            error: The error to handle

        Returns:
            List of applicable recovery actions
        """
        handlers = []
        for error_type, actions in self._recovery_handlers.items():
            if isinstance(error, error_type):
                handlers.extend(actions)
        return handlers

    def handle_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Handle an error using registered recovery strategies.

        Args:
            error: The error to handle
            context: Additional context information

        Returns:
            Result of recovery action if successful, None otherwise

        Raises:
            Exception: If no recovery strategy succeeds
        """
        handlers = self.get_handlers(error)
        if not handlers:
            raise error

        for handler in handlers:
            try:
                if handler.cleanup_func:
                    handler.cleanup_func()

                if handler.strategy == RecoveryStrategy.RETRY:
                    return self._handle_retry(error, handler, context)
                elif handler.strategy == RecoveryStrategy.FALLBACK:
                    return self._handle_fallback(error, handler, context)
                elif handler.strategy == RecoveryStrategy.SKIP:
                    return self._handle_skip(error, handler, context)
                elif handler.strategy == RecoveryStrategy.ABORT:
                    return self._handle_abort(error, handler, context)

            except Exception as e:
                self.logger.warning(
                    f"Recovery handler failed: {str(e)}",
                    extra={"original_error": str(error), "handler": str(handler)},
                )
                continue

        raise error

    def _handle_retry(
        self,
        error: Exception,
        handler: RecoveryAction,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Handle retry strategy."""
        if not hasattr(error, "retry_count"):
            error.retry_count = 0  # type: ignore

        if error.retry_count >= handler.max_retries:  # type: ignore
            raise error

        error.retry_count += 1  # type: ignore
        time.sleep(handler.retry_delay)

        self.logger.info(
            f"Retrying operation (attempt {error.retry_count}/{handler.max_retries})",  # type: ignore
            extra={"error": str(error), "context": context},
        )

        # The actual retry happens in the recovery decorator
        return None

    def _handle_fallback(
        self,
        error: Exception,
        handler: RecoveryAction,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Handle fallback strategy."""
        self.logger.info(
            "Using fallback value",
            extra={
                "error": str(error),
                "fallback_value": str(handler.fallback_value),
                "context": context,
            },
        )
        return handler.fallback_value

    def _handle_skip(
        self,
        error: Exception,
        handler: RecoveryAction,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Handle skip strategy."""
        self.logger.info(
            "Skipping operation", extra={"error": str(error), "context": context}
        )
        return None

    def _handle_abort(
        self,
        error: Exception,
        handler: RecoveryAction,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Handle abort strategy."""
        self.logger.error(
            "Aborting operation", extra={"error": str(error), "context": context}
        )
        raise error


# Global recovery manager instance
_default_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """
    Get or create the global recovery manager instance.

    Returns:
        The global RecoveryManager instance
    """
    global _default_recovery_manager

    if _default_recovery_manager is None:
        _default_recovery_manager = RecoveryManager()

    return _default_recovery_manager


def recoverable(
    error_handlers: Optional[Dict[type, RecoveryAction]] = None,
    context_name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for making functions recoverable from errors.

    Args:
        error_handlers: Dictionary mapping error types to recovery actions
        context_name: Name for the error context

    Returns:
        Decorated function

    Example:
        >>> @recoverable({ValueError: RecoveryAction(RecoveryStrategy.RETRY)})
        ... def process_data(data):
        ...     # Process data
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            manager = get_recovery_manager()

            # Register handlers if provided
            if error_handlers:
                for error_type, action in error_handlers.items():
                    manager.register_handler(
                        error_type,
                        action.strategy,
                        action.max_retries,
                        action.retry_delay,
                        action.fallback_value,
                        action.cleanup_func,
                    )

            ctx_name = context_name or func.__name__
            with error_context(ctx_name) as ctx:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    ctx.additional_info["error"] = str(e)
                    result = manager.handle_error(e, {"args": args, "kwargs": kwargs})
                    if result is not None:
                        return cast(T, result)
                    # If result is None and it's a retry strategy,
                    # try the function again
                    if (
                        isinstance(e, Exception)
                        and hasattr(e, "retry_count")
                        and e.retry_count < getattr(e, "max_retries", float("inf"))  # type: ignore
                    ):
                        return func(*args, **kwargs)
                    raise

        return wrapper

    return decorator


@contextmanager
def recovery_context(
    name: str, error_handlers: Optional[Dict[type, RecoveryAction]] = None
) -> Generator[None, None, None]:
    """
    Context manager for error recovery.

    Args:
        name: Name for the recovery context
        error_handlers: Dictionary mapping error types to recovery actions

    Example:
        >>> with recovery_context("data_processing", {
        ...     ValueError: RecoveryAction(RecoveryStrategy.RETRY)
        ... }):
        ...     process_data()
    """
    manager = get_recovery_manager()

    # Register handlers if provided
    if error_handlers:
        for error_type, action in error_handlers.items():
            manager.register_handler(
                error_type,
                action.strategy,
                action.max_retries,
                action.retry_delay,
                action.fallback_value,
                action.cleanup_func,
            )

    with error_context(name) as ctx:
        try:
            yield
        except Exception as e:
            ctx.additional_info["error"] = str(e)
            result = manager.handle_error(e, {"context_name": name})
            if result is not None:
                return
            # If result is None and it's a retry strategy,
            # let the exception propagate to retry the context
            raise
