"""Decorators for runtime type checking and validation.

This module provides decorators for runtime type checking and parameter validation
to ensure type safety and proper parameter usage at runtime.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints

from .types import ValidationResult

F = TypeVar("F", bound=Callable[..., Any])


def type_check(func: F) -> F:
    """Decorator for runtime type checking of function arguments and return value.

    This decorator checks that the types of arguments passed to the function
    match their type hints, and that the return value matches its type hint.

    Parameters
    ----------
    func : Callable
        The function to decorate

    Returns
    -------
    Callable
        The decorated function

    Raises
    ------
    TypeError
        If an argument or return value has an incorrect type
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Combine positional and keyword arguments
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Check argument types
        for name, value in bound_args.arguments.items():
            if name in hints:
                expected_type = hints[name]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' must be of type {expected_type}, "
                        f"got {type(value)}"
                    )

        # Call function
        result = func(*args, **kwargs)

        # Check return type
        if "return" in hints:
            expected_type = hints["return"]
            if not isinstance(result, expected_type):
                raise TypeError(
                    f"Return value must be of type {expected_type}, "
                    f"got {type(result)}"
                )

        return result

    return wrapper  # type: ignore


def validate_params(
    validator: Optional[Callable[[Dict[str, Any]], ValidationResult]] = None,
    **param_types: Type[Any],
) -> Callable[[F], F]:
    """Decorator for parameter validation.

    This decorator validates function parameters using either a custom validator
    function or type specifications.

    Parameters
    ----------
    validator : Optional[Callable[[Dict[str, Any]], ValidationResult]]
        Optional custom validator function
    **param_types : Type[Any]
        Type specifications for parameters

    Returns
    -------
    Callable[[F], F]
        Decorator function

    Examples
    --------
    >>> @validate_params(n_clusters=int, tol=float)
    ... def cluster(n_clusters: int, tol: float = 0.001) -> None:
    ...     pass

    >>> @validate_params(validator=my_validator)
    ... def optimize(params: Dict[str, Any]) -> None:
    ...     pass
    """

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Combine positional and keyword arguments
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Get parameter values
            params = bound_args.arguments

            # Run custom validator if provided
            if validator is not None:
                result = validator(params)
                if not result.is_valid:
                    raise ValueError(
                        "Parameter validation failed:\n"
                        + "\n".join(f"- {error}" for error in result.errors)
                    )

            # Check parameter types
            for name, expected_type in param_types.items():
                if name in params:
                    value = params[name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{name}' must be of type {expected_type}, "
                            f"got {type(value)}"
                        )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
