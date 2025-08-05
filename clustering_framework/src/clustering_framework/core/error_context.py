"""
Error context management for the clustering framework.

This module provides utilities for managing and enriching error context information,
making it easier to debug and understand errors when they occur.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ErrorContext:
    """Container for error context information."""

    timestamp: datetime = field(default_factory=datetime.now)
    function_name: str = ""
    module_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional["ErrorContext"] = None


class ContextManager:
    """Manages error context information throughout the application."""

    _context = threading.local()

    @classmethod
    def get_current_context(cls) -> Optional[ErrorContext]:
        """Get the current error context."""
        return getattr(cls._context, "current", None)

    @classmethod
    def set_context(cls, context: ErrorContext) -> None:
        """Set the current error context."""
        cls._context.current = context

    @classmethod
    def clear_context(cls) -> None:
        """Clear the current error context."""
        if hasattr(cls._context, "current"):
            delattr(cls._context, "current")


@contextmanager
def error_context(
    function_name: str = "", module_name: str = "", **kwargs: Any
) -> Generator[ErrorContext, None, None]:
    """
    Context manager for handling error context information.

    Args:
        function_name: Name of the function where the context is created
        module_name: Name of the module where the context is created
        **kwargs: Additional context parameters

    Yields:
        ErrorContext: The current error context object

    Example:
        >>> with error_context(function_name="process_data", batch_size=100) as ctx:
        ...     # Do something that might raise an error
        ...     ctx.additional_info['progress'] = 50
    """
    parent_context = ContextManager.get_current_context()
    current_context = ErrorContext(
        function_name=function_name,
        module_name=module_name,
        parameters=kwargs,
        parent_context=parent_context,
    )

    ContextManager.set_context(current_context)

    try:
        yield current_context
    finally:
        ContextManager.set_context(parent_context)


def get_full_context() -> Dict[str, Any]:
    """
    Get the complete error context chain.

    Returns:
        Dict containing the full context chain information
    """
    context = ContextManager.get_current_context()
    if not context:
        return {}

    context_chain = []
    current = context

    while current:
        context_info = {
            "timestamp": current.timestamp.isoformat(),
            "function_name": current.function_name,
            "module_name": current.module_name,
            "parameters": current.parameters,
            "additional_info": current.additional_info,
        }
        context_chain.append(context_info)
        current = current.parent_context

    return {"context_chain": context_chain, "context_depth": len(context_chain)}


def enrich_exception(
    exc: Exception, additional_info: Optional[Dict[str, Any]] = None
) -> Exception:
    """
    Enrich an exception with current context information.

    Args:
        exc: The exception to enrich
        additional_info: Additional information to add to the context

    Returns:
        The enriched exception
    """
    from .exceptions import ClusteringError

    if isinstance(exc, ClusteringError):
        context = get_full_context()
        if additional_info:
            context["additional_info"] = additional_info
        exc.context.update(context)

    return exc
