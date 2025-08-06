"""Lazy loading functionality for clustering algorithms with instance pooling."""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Set, Generator
from weakref import WeakSet
import time
from dataclasses import dataclass
from contextlib import contextmanager

from clustering_framework.algorithms import ClusteringAlgorithm

T = TypeVar("T", bound=ClusteringAlgorithm)

__all__ = ["LazyLoader", "InstanceMetadata"]


@dataclass
class InstanceMetadata:
    """Metadata for algorithm instances."""

    created_at: float
    last_used: float
    use_count: int


class LazyLoader:
    """Lazy loader for clustering algorithms with instance pooling.

    This class provides lazy loading functionality for clustering algorithms,
    deferring instance creation until actually needed. It also handles
    parameter validation, instance pooling, lifecycle management, and cleanup.

    Features:
    - Lazy instantiation of algorithm instances
    - Parameter validation and caching
    - Instance pooling with lifecycle management
    - Automatic cleanup of unused instances
    - Usage tracking and statistics

    Example
    -------
    >>> loader = LazyLoader(KMeansAlgorithm)
    >>> with loader.get_instance(random_state=42) as instance:
    ...     print(instance.name)
    """

    def __init__(
        self,
        algorithm_class: Type[T],
        validation_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_pool_size: int = 10,
        instance_ttl: float = 300.0,  # 5 minutes
    ):
        """Initialize the lazy loader.

        Parameters
        ----------
        algorithm_class : Type[ClusteringAlgorithm]
            The algorithm class to lazily instantiate
        validation_hook : callable, optional
            Optional function to perform additional validation on parameters
        max_pool_size : int, optional
            Maximum number of instances to keep in the pool (default: 10)
        instance_ttl : float, optional
            Time-to-live for unused instances in seconds (default: 300.0)
        """
        self._algorithm_class = algorithm_class
        self._validation_hook = validation_hook
        self._validated_params: Dict[int, Dict[str, Any]] = {}
        self._validation_cache: Dict[str, bool] = {}

        # Instance pooling
        self._instance_pool: Dict[int, T] = {}
        self._instance_metadata: Dict[int, InstanceMetadata] = {}
        self._max_pool_size = max_pool_size
        self._instance_ttl = instance_ttl

        # Active instances tracking
        self._active_instances: WeakSet[T] = WeakSet()

        # Cleanup hook registry
        self._cleanup_hooks: Set[Callable[[T], None]] = set()

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
            True if parameters are valid
        """
        # Create cache key from params
        cache_key = str(sorted(params.items()))

        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        try:
            # Create temporary instance for validation
            temp_instance = self._algorithm_class()

            # Get default parameters and merge with provided ones
            default_params = temp_instance.get_default_parameters()
            merged_params = default_params.copy()
            merged_params.update(params)

            # Validate merged parameters
            temp_instance.validate_parameters(merged_params)

            # Run additional validation if provided
            if self._validation_hook is not None:
                self._validation_hook(merged_params)

            self._validation_cache[cache_key] = True
            return True
        except (ValueError, TypeError):
            self._validation_cache[cache_key] = False
            return False

    def register_cleanup_hook(self, hook: Callable[[T], None]) -> None:
        """Register a cleanup hook to be called when instances are removed.

        Parameters
        ----------
        hook : callable
            Function to be called with the instance being cleaned up
        """
        self._cleanup_hooks.add(hook)

    def _cleanup_instance(self, instance: T) -> None:
        """Run cleanup hooks and remove instance from tracking.

        Parameters
        ----------
        instance : ClusteringAlgorithm
            The instance to clean up
        """
        for hook in self._cleanup_hooks:
            try:
                hook(instance)
            except Exception:
                # Log error but continue with other hooks
                pass

        self._active_instances.discard(instance)

    def _cleanup_pool(self) -> None:
        """Clean up expired instances from the pool."""
        current_time = time.time()
        expired_keys = []

        for random_state, metadata in self._instance_metadata.items():
            if (current_time - metadata.last_used) > self._instance_ttl:
                expired_keys.append(random_state)

        for key in expired_keys:
            instance = self._instance_pool.pop(key, None)
            if instance is not None:
                self._cleanup_instance(instance)
            self._instance_metadata.pop(key, None)

    @contextmanager
    def get_instance(
        self, random_state: Optional[int] = None, **kwargs: Any
    ) -> Generator[ClusteringAlgorithm, None, None]:
        """Get or create an algorithm instance using the instance pool.

        This is a context manager that ensures proper instance lifecycle
        management and cleanup.

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
        # Get default parameters and merge with provided ones
        temp_instance = self._algorithm_class()
        default_params = temp_instance.get_default_parameters()
        merged_params = default_params.copy()
        merged_params.update(kwargs)

        # Validate parameters
        if not self.validate_parameters(merged_params):
            raise ValueError("Invalid parameters for algorithm")

        # Clean up expired instances
        self._cleanup_pool()

        # Use random_state as instance key if provided
        instance_key = random_state if random_state is not None else id(merged_params)

        try:
            # Try to get instance from pool
            if instance_key in self._instance_pool:
                instance = self._instance_pool[instance_key]
                metadata = self._instance_metadata[instance_key]
                metadata.last_used = time.time()
                metadata.use_count += 1
            else:
                # Create new instance if not in pool
                instance = self._algorithm_class(random_state=random_state)
                instance.set_parameters(merged_params)

                # Add to pool if not at capacity
                if len(self._instance_pool) < self._max_pool_size:
                    self._instance_pool[instance_key] = instance
                    self._instance_metadata[instance_key] = InstanceMetadata(
                        created_at=time.time(), last_used=time.time(), use_count=1
                    )

            # Track active instance
            self._active_instances.add(instance)

            # Store validated parameters
            if random_state is not None:
                self._validated_params[random_state] = merged_params.copy()

            yield instance

        finally:
            # Update last used time and remove from active instances
            if instance_key in self._instance_metadata:
                self._instance_metadata[instance_key].last_used = time.time()
            self._active_instances.discard(instance)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the instance pool.

        Returns
        -------
        dict
            Statistics about the instance pool including:
            - pool_size: current number of instances in pool
            - active_instances: number of currently active instances
            - total_uses: total number of instance uses
            - oldest_instance_age: age of oldest instance in seconds
        """
        current_time = time.time()
        stats = {
            "pool_size": len(self._instance_pool),
            "active_instances": len(self._active_instances),
            "total_uses": sum(
                meta.use_count for meta in self._instance_metadata.values()
            ),
            "oldest_instance_age": 0.0,
        }

        if self._instance_metadata:
            oldest_time = min(
                meta.created_at for meta in self._instance_metadata.values()
            )
            stats["oldest_instance_age"] = current_time - oldest_time

        return stats

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
