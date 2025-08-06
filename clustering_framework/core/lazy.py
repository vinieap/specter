"""Lazy loading functionality for clustering algorithms."""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from ..algorithms import ClusteringAlgorithm

T = TypeVar("T", bound=ClusteringAlgorithm)


class LazyLoader:
    """Lazy loader for clustering algorithms.

    This class provides lazy loading functionality for clustering algorithms,
    deferring instance creation until actually needed. It also handles
    parameter validation and caching of validation results.

    Example
    -------
    >>> loader = LazyLoader(KMeansAlgorithm)
    >>> instance = loader.get_instance(random_state=42)
    >>> print(instance.name)
    """

    def __init__(
        self,
        algorithm_class: Type[T],
        validation_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize the lazy loader.

        Parameters
        ----------
        algorithm_class : Type[ClusteringAlgorithm]
            The algorithm class to lazily instantiate
        validation_hook : callable, optional
            Optional function to perform additional validation on parameters
        """
        self._algorithm_class = algorithm_class
        self._validation_hook = validation_hook
        self._validated_params: Dict[int, Dict[str, Any]] = {}
        self._validation_cache: Dict[str, bool] = {}

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameters without creating an instance.

        This method validates parameters using the algorithm's validation
        logic and caches the results to avoid redundant validation.

        Parameters
        ----------
        params : dict
            Parameters to validate

        Returns
        -------
        bool
            True if parameters are valid, False otherwise
        """
        # Create cache key from params
        cache_key = str(sorted(params.items()))

        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        try:
            # Create temporary instance for validation
            temp_instance = self._algorithm_class()
            temp_instance.validate_parameters(params)

            # Run additional validation if provided
            if self._validation_hook is not None:
                self._validation_hook(params)

            self._validation_cache[cache_key] = True
            return True
        except (ValueError, TypeError):
            self._validation_cache[cache_key] = False
            return False

    def get_instance(
        self, random_state: Optional[int] = None, **kwargs: Any
    ) -> ClusteringAlgorithm:
        """Get or create an algorithm instance.

        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility
        **kwargs : dict
            Additional parameters for the algorithm

        Returns
        -------
        ClusteringAlgorithm
            The algorithm instance

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # Validate parameters first
        if not self.validate_parameters(kwargs):
            raise ValueError("Invalid parameters for algorithm")

        # Create instance with validated parameters
        instance = self._algorithm_class(random_state=random_state)

        # Store validated parameters
        if random_state is not None:
            self._validated_params[random_state] = kwargs.copy()

        return instance

    def clear_validation_cache(self) -> None:
        """Clear the parameter validation cache."""
        self._validation_cache.clear()

    def get_validated_params(self, random_state: int) -> Optional[Dict[str, Any]]:
        """Get previously validated parameters for a random state.

        Parameters
        ----------
        random_state : int
            The random state to get parameters for

        Returns
        -------
        dict or None
            The validated parameters if found, None otherwise
        """
        return self._validated_params.get(random_state)
