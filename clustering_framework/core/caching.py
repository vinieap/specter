"""Enhanced caching system for clustering algorithms."""

from collections import OrderedDict
from typing import Any, Dict, Generic, Optional, TypeVar
from dataclasses import dataclass
import time

T = TypeVar("T")


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    last_cleanup: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """LRU cache implementation with monitoring.

    This class implements a Least Recently Used (LRU) cache with
    built-in monitoring and statistics. It provides efficient
    instance management and automatic cleanup.

    Example
    -------
    >>> cache = LRUCache[ClusteringAlgorithm](max_size=10)
    >>> cache.put("key1", instance1)
    >>> instance = cache.get("key1")
    """

    def __init__(self, max_size: int = 10, cleanup_threshold: float = 0.8):
        """Initialize the LRU cache.

        Parameters
        ----------
        max_size : int, default=10
            Maximum number of items to store
        cleanup_threshold : float, default=0.8
            Fraction of max_size at which to trigger cleanup
        """
        self._cache: OrderedDict[Any, T] = OrderedDict()
        self._max_size = max_size
        self._cleanup_threshold = cleanup_threshold
        self._stats = CacheStats()

    def get(self, key: Any) -> Optional[T]:
        """Get an item from the cache.

        Parameters
        ----------
        key : Any
            The key to look up

        Returns
        -------
        T or None
            The cached item if found, None otherwise
        """
        if key in self._cache:
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._stats.hits += 1
            return value

        self._stats.misses += 1
        return None

    def put(self, key: Any, value: T) -> None:
        """Add an item to the cache.

        Parameters
        ----------
        key : Any
            The key for the item
        value : T
            The item to cache
        """
        if key in self._cache:
            # Update existing entry
            self._cache.pop(key)
        elif len(self._cache) >= self._max_size:
            # Remove oldest item
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        self._cache[key] = value
        self._stats.total_size = len(self._cache)

        # Check if cleanup is needed
        if self._should_cleanup():
            self.cleanup()

    def _should_cleanup(self) -> bool:
        """Check if cache cleanup should be triggered."""
        return len(self._cache) >= int(self._max_size * self._cleanup_threshold)

    def cleanup(self) -> None:
        """Perform cache cleanup.

        This method removes the oldest items from the cache until
        it's below the cleanup threshold.
        """
        target_size = int(self._max_size * 0.5)  # Reduce to 50% capacity
        while len(self._cache) > target_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        self._stats.last_cleanup = time.time()
        self._stats.total_size = len(self._cache)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns
        -------
        CacheStats
            Current cache statistics
        """
        return self._stats

    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        return key in self._cache


class CacheManager:
    """Manager for multiple algorithm caches.

    This class manages multiple LRU caches, one for each algorithm type,
    and provides centralized cache management and monitoring.

    Example
    -------
    >>> manager = CacheManager()
    >>> manager.get_cache("kmeans").put(key, instance)
    >>> instance = manager.get_cache("kmeans").get(key)
    """

    def __init__(self, default_size: int = 10):
        """Initialize the cache manager.

        Parameters
        ----------
        default_size : int, default=10
            Default size for new caches
        """
        self._caches: Dict[str, LRUCache[Any]] = {}
        self._default_size = default_size

    def get_cache(self, name: str, max_size: Optional[int] = None) -> LRUCache[Any]:
        """Get or create a cache for an algorithm.

        Parameters
        ----------
        name : str
            Algorithm name
        max_size : int, optional
            Maximum cache size. Uses default_size if not specified.

        Returns
        -------
        LRUCache
            The cache instance for the algorithm
        """
        if name not in self._caches:
            self._caches[name] = LRUCache(
                max_size=max_size if max_size is not None else self._default_size
            )
        return self._caches[name]

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches.

        Returns
        -------
        dict
            Dictionary mapping algorithm names to their cache statistics
        """
        return {name: cache.stats for name, cache in self._caches.items()}
