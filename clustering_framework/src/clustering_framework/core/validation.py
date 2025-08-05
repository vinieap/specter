"""Parameter validation base class and utilities.

This module provides a flexible and extensible base class for parameter validation
along with common validation methods and utilities that can be used across the framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Generic
from enum import Enum
import numbers


# Type variable for generic validation
T = TypeVar("T")


class ValidationErrorType(Enum):
    """Types of validation errors."""

    MISSING_PARAMETER = "missing_parameter"
    INVALID_TYPE = "invalid_type"
    INVALID_RANGE = "invalid_range"
    INVALID_CHOICE = "invalid_choice"
    INVALID_DEPENDENCY = "invalid_dependency"
    CUSTOM = "custom"


@dataclass
class ValidationError:
    """Represents a single validation error."""

    error_type: ValidationErrorType
    parameter: str
    message: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Contains the results of a validation operation."""

    is_valid: bool
    errors: List[ValidationError]
    context: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:
        return self.is_valid


class ValidationContext:
    """Maintains context during validation operations."""

    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._errors: List[ValidationError] = []

    def add_context(self, key: str, value: Any) -> None:
        """Add information to the validation context."""
        self._context[key] = value

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        self._errors.append(error)

    def get_context(self, key: str) -> Any:
        """Get context value by key."""
        return self._context.get(key)

    def get_result(self) -> ValidationResult:
        """Get the validation result."""
        return ValidationResult(
            is_valid=len(self._errors) == 0,
            errors=self._errors.copy(),
            context=self._context.copy(),
        )


class BaseValidator(ABC, Generic[T]):
    """Base class for parameter validation.

    This class provides a foundation for implementing parameter validation
    with support for type checking, range validation, categorical parameters,
    and custom validation rules.

    Parameters
    ----------
    required_params : Set[str]
        Set of parameter names that must be present
    """

    def __init__(self, required_params: Optional[Set[str]] = None):
        self.required_params = required_params or set()
        self._context = ValidationContext()

    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate parameters according to defined rules.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate

        Returns
        -------
        ValidationResult
            Result of validation containing any errors found
        """
        self._context = ValidationContext()

        # Check required parameters
        self._validate_required_params(params)

        # Run type validation
        self._validate_types(params)

        # Run range validation
        self._validate_ranges(params)

        # Run categorical validation
        self._validate_categorical(params)

        # Run dependency validation
        self._validate_dependencies(params)

        # Run custom validation
        self._validate_custom(params)

        return self._context.get_result()

    def _validate_required_params(self, params: Dict[str, Any]) -> None:
        """Check that all required parameters are present."""
        missing = self.required_params - set(params.keys())
        if missing:
            self._context.add_error(
                ValidationError(
                    error_type=ValidationErrorType.MISSING_PARAMETER,
                    parameter=", ".join(sorted(missing)),
                    message=f"Missing required parameters: {missing}",
                )
            )

    def _validate_types(self, params: Dict[str, Any]) -> None:
        """Validate parameter types."""
        type_specs = self.get_type_specs()
        for param, value in params.items():
            if param in type_specs:
                expected_type = type_specs[param]
                if not isinstance(value, expected_type):
                    self._context.add_error(
                        ValidationError(
                            error_type=ValidationErrorType.INVALID_TYPE,
                            parameter=param,
                            message=f"Parameter {param} must be of type {expected_type.__name__}, got {type(value).__name__}",
                        )
                    )

    def _validate_ranges(self, params: Dict[str, Any]) -> None:
        """Validate numeric parameter ranges."""
        range_specs = self.get_range_specs()
        for param, value in params.items():
            if param in range_specs:
                min_val, max_val = range_specs[param]
                if isinstance(value, numbers.Number):
                    if min_val is not None and value < min_val:
                        self._context.add_error(
                            ValidationError(
                                error_type=ValidationErrorType.INVALID_RANGE,
                                parameter=param,
                                message=f"Parameter {param} must be >= {min_val}, got {value}",
                            )
                        )
                    if max_val is not None and value > max_val:
                        self._context.add_error(
                            ValidationError(
                                error_type=ValidationErrorType.INVALID_RANGE,
                                parameter=param,
                                message=f"Parameter {param} must be <= {max_val}, got {value}",
                            )
                        )

    def _validate_categorical(self, params: Dict[str, Any]) -> None:
        """Validate categorical parameters."""
        categorical_specs = self.get_categorical_specs()
        for param, value in params.items():
            if param in categorical_specs:
                valid_values = categorical_specs[param]
                if value not in valid_values:
                    self._context.add_error(
                        ValidationError(
                            error_type=ValidationErrorType.INVALID_CHOICE,
                            parameter=param,
                            message=f"Parameter {param} must be one of {valid_values}, got {value}",
                        )
                    )

    def _validate_dependencies(self, params: Dict[str, Any]) -> None:
        """Validate parameter dependencies."""
        dependency_specs = self.get_dependency_specs()
        for param, dependencies in dependency_specs.items():
            if param in params:
                for dep_param, valid_values in dependencies.items():
                    if dep_param not in params:
                        self._context.add_error(
                            ValidationError(
                                error_type=ValidationErrorType.INVALID_DEPENDENCY,
                                parameter=param,
                                message=f"Parameter {param} requires {dep_param} to be set",
                            )
                        )
                    elif params[dep_param] not in valid_values:
                        self._context.add_error(
                            ValidationError(
                                error_type=ValidationErrorType.INVALID_DEPENDENCY,
                                parameter=param,
                                message=f"Parameter {param} requires {dep_param} to be one of {valid_values}",
                            )
                        )

    def _validate_custom(self, params: Dict[str, Any]) -> None:
        """Run custom validation rules."""
        try:
            self.validate_custom(params)
        except Exception as e:
            self._context.add_error(
                ValidationError(
                    error_type=ValidationErrorType.CUSTOM,
                    parameter="custom",
                    message=str(e),
                )
            )

    @abstractmethod
    def get_type_specs(self) -> Dict[str, Type]:
        """Get type specifications for parameters.

        Returns
        -------
        Dict[str, Type]
            Mapping of parameter names to their expected types
        """
        return {}

    @abstractmethod
    def get_range_specs(self) -> Dict[str, tuple[Optional[float], Optional[float]]]:
        """Get range specifications for numeric parameters.

        Returns
        -------
        Dict[str, tuple[Optional[float], Optional[float]]]
            Mapping of parameter names to (min, max) tuples
        """
        return {}

    @abstractmethod
    def get_categorical_specs(self) -> Dict[str, Set[Any]]:
        """Get specifications for categorical parameters.

        Returns
        -------
        Dict[str, Set[Any]]
            Mapping of parameter names to sets of valid values
        """
        return {}

    @abstractmethod
    def get_dependency_specs(self) -> Dict[str, Dict[str, Set[Any]]]:
        """Get parameter dependency specifications.

        Returns
        -------
        Dict[str, Dict[str, Set[Any]]]
            Mapping of parameters to their dependencies
        """
        return {}

    def validate_custom(self, params: Dict[str, Any]) -> None:
        """Implement custom validation rules.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate

        Raises
        ------
        Exception
            If validation fails
        """
        pass
