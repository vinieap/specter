"""Dependency injection system for managing component dependencies."""

from typing import Dict, Any, Type, TypeVar, cast

T = TypeVar("T")


class DependencyContainer:
    """Container for managing dependencies.

    This class provides a simple dependency injection container that helps
    manage component dependencies and reduce coupling between modules.

    Example
    -------
    >>> container = DependencyContainer()
    >>> container.register(Logger, ConsoleLogger)
    >>> logger = container.resolve(Logger)
    """

    def __init__(self):
        """Initialize the dependency container."""
        self._bindings: Dict[Type, Type] = {}
        self._instances: Dict[Type, Any] = {}

    def register(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register an implementation for an interface.

        Parameters
        ----------
        interface : Type[T]
            The interface or base class
        implementation : Type[T]
            The concrete implementation class

        Raises
        ------
        ValueError
            If the implementation doesn't inherit from the interface
        """
        if not issubclass(implementation, interface):
            raise ValueError(
                f"{implementation.__name__} must inherit from {interface.__name__}"
            )
        self._bindings[interface] = implementation

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a singleton instance.

        Parameters
        ----------
        interface : Type[T]
            The interface or base class
        instance : T
            The instance to register

        Raises
        ------
        ValueError
            If the instance is not of the correct type
        """
        if not isinstance(instance, interface):
            raise ValueError(f"Instance must be of type {interface.__name__}")
        self._instances[interface] = instance

    def resolve(self, interface: Type[T]) -> T:
        """Resolve an implementation for an interface.

        Parameters
        ----------
        interface : Type[T]
            The interface or base class to resolve

        Returns
        -------
        T
            An instance of the implementation

        Raises
        ------
        KeyError
            If no implementation is registered for the interface
        """
        # Check for singleton instance
        if interface in self._instances:
            return cast(T, self._instances[interface])

        # Check for registered implementation
        if interface not in self._bindings:
            raise KeyError(f"No implementation registered for {interface.__name__}")

        # Create new instance
        implementation = self._bindings[interface]
        instance = implementation()
        return cast(T, instance)

    def clear(self) -> None:
        """Clear all registrations."""
        self._bindings.clear()
        self._instances.clear()


# Global container instance
container = DependencyContainer()
