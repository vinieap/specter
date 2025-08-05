"""Registry system for clustering algorithms."""

from typing import Dict, Type, List, Optional

from ..algorithms import ClusteringAlgorithm


class AlgorithmRegistry:
    """Registry for managing available clustering algorithms.

    This class provides a central registry for all clustering algorithms,
    allowing algorithms to be registered, retrieved, and organized by
    category. It ensures that algorithms are properly initialized and
    maintains metadata about available algorithms.

    Example
    -------
    >>> registry = AlgorithmRegistry()
    >>> registry.register_algorithm(KMeansAlgorithm)
    >>> algo = registry.get_algorithm("kmeans")
    >>> print(registry.list_algorithms())
    """

    def __init__(self, register_defaults: bool = True):
        """Initialize the registry.

        Parameters
        ----------
        register_defaults : bool, default=True
            Whether to register default algorithms during initialization
        """
        self._algorithms: Dict[str, Type[ClusteringAlgorithm]] = {}
        self._instance_cache: Dict[str, Dict[Optional[int], ClusteringAlgorithm]] = {}
        self._max_cache_size = 10  # Maximum instances per algorithm
        if register_defaults:
            self._register_default_algorithms()

    def _register_default_algorithms(self):
        """Register the default set of clustering algorithms."""
        from ..algorithms import (
            KMeansAlgorithm,
            MiniBatchKMeansAlgorithm,
            BisectingKMeansAlgorithm,
            DBSCANAlgorithm,
            HDBSCANAlgorithm,
            OPTICSAlgorithm,
            AgglomerativeClusteringAlgorithm,
            BirchAlgorithm,
            AffinityPropagationAlgorithm,
            SpectralClusteringAlgorithm,
            MeanShiftAlgorithm,
        )

        # Basic clustering algorithms
        self.register_algorithm(KMeansAlgorithm)
        self.register_algorithm(MiniBatchKMeansAlgorithm)
        self.register_algorithm(BisectingKMeansAlgorithm)

        # Density-based clustering
        self.register_algorithm(DBSCANAlgorithm)
        self.register_algorithm(HDBSCANAlgorithm)
        self.register_algorithm(OPTICSAlgorithm)

        # Hierarchical clustering
        self.register_algorithm(AgglomerativeClusteringAlgorithm)
        self.register_algorithm(BirchAlgorithm)

        # Affinity/similarity-based clustering
        self.register_algorithm(AffinityPropagationAlgorithm)
        self.register_algorithm(SpectralClusteringAlgorithm)
        self.register_algorithm(MeanShiftAlgorithm)

    def _get_algorithm_name(self, algorithm_class: Type[ClusteringAlgorithm]) -> str:
        """Get the name of an algorithm class without instantiating it.

        Parameters
        ----------
        algorithm_class : Type[ClusteringAlgorithm]
            The algorithm class to get the name from

        Returns
        -------
        str
            The lowercase algorithm name
        """
        # Try to get name from class attribute first
        if hasattr(algorithm_class, "_algorithm_name"):
            return algorithm_class._algorithm_name.lower()

        # Create temporary instance as fallback
        temp_instance = algorithm_class()
        name = temp_instance.name.lower()

        # Cache the name on the class for future use
        algorithm_class._algorithm_name = name
        return name

    def register_algorithm(self, algorithm_class: Type[ClusteringAlgorithm]) -> None:
        """Register a new clustering algorithm.

        Parameters
        ----------
        algorithm_class : Type[ClusteringAlgorithm]
            The clustering algorithm class to register

        Raises
        ------
        ValueError
            If algorithm with same name is already registered
        TypeError
            If algorithm_class doesn't inherit from ClusteringAlgorithm
        """
        if not issubclass(algorithm_class, ClusteringAlgorithm):
            raise TypeError(
                f"Algorithm class must inherit from ClusteringAlgorithm, "
                f"got {algorithm_class.__name__}"
            )

        name = self._get_algorithm_name(algorithm_class)

        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' is already registered")

        self._algorithms[name] = algorithm_class

    def _cleanup_cache(self, name: str) -> None:
        """Clean up old instances from cache if needed.

        Parameters
        ----------
        name : str
            Algorithm name whose cache to clean
        """
        if name not in self._instance_cache:
            return

        cache = self._instance_cache[name]
        if len(cache) > self._max_cache_size:
            # Remove oldest instances (we could use LRU but keeping it simple)
            while len(cache) > self._max_cache_size:
                # Get first key and remove it
                key = next(iter(cache))
                instance = cache.pop(key)
                # Call cleanup hook if available
                if hasattr(instance, "cleanup"):
                    instance.cleanup()

    def get_algorithm(
        self, name: str, random_state: Optional[int] = None
    ) -> ClusteringAlgorithm:
        """Get an instance of a clustering algorithm by name.

        Parameters
        ----------
        name : str
            Name of the algorithm to get
        random_state : int, optional
            Random state for reproducibility

        Returns
        -------
        ClusteringAlgorithm
            Instance of the requested algorithm

        Raises
        ------
        ValueError
            If algorithm name is not registered
        """
        name = name.lower()
        algorithm_class = self._algorithms.get(name)
        if algorithm_class is None:
            available = list(self._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm: {name}. Available algorithms: {available}"
            )

        # Check cache first
        if name in self._instance_cache:
            if random_state in self._instance_cache[name]:
                return self._instance_cache[name][random_state]

        # Create new instance
        instance = algorithm_class(random_state=random_state)

        # Cache the instance
        if name not in self._instance_cache:
            self._instance_cache[name] = {}
        self._instance_cache[name][random_state] = instance

        # Cleanup if needed
        self._cleanup_cache(name)

        return instance

    def list_algorithms(self) -> List[str]:
        """Get a list of all registered algorithm names.

        Returns
        -------
        List[str]
            List of algorithm names
        """
        return sorted(self._algorithms.keys())

    def _get_algorithm_category(
        self, algorithm_class: Type[ClusteringAlgorithm]
    ) -> str:
        """Get the category of an algorithm class without instantiating it.

        Parameters
        ----------
        algorithm_class : Type[ClusteringAlgorithm]
            The algorithm class to get the category from

        Returns
        -------
        str
            The category name
        """
        # Try to get category from class attribute first
        if hasattr(algorithm_class, "_algorithm_category"):
            return algorithm_class._algorithm_category

        # Create temporary instance as fallback
        temp_instance = algorithm_class()
        category = temp_instance.category

        # Cache the category on the class for future use
        algorithm_class._algorithm_category = category
        return category

    def get_algorithm_categories(self) -> Dict[str, List[str]]:
        """Get algorithms grouped by category.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping category names to lists of algorithm names
        """
        categories: Dict[str, List[str]] = {
            category: [] for category in ClusteringAlgorithm.CATEGORIES.values()
        }

        for name, algo_class in self._algorithms.items():
            category = self._get_algorithm_category(algo_class)
            category_name = ClusteringAlgorithm.CATEGORIES[category]
            categories[category_name].append(name)

        # Sort algorithms within each category
        for category in categories:
            categories[category].sort()

        return categories

    def __str__(self) -> str:
        """String representation showing registered algorithms by category.

        Returns
        -------
        str
            Formatted string of algorithms by category
        """
        categories = self.get_algorithm_categories()
        lines = ["Available Clustering Algorithms:"]

        for category, algorithms in categories.items():
            if algorithms:  # Only show categories with algorithms
                lines.append(f"\n{category}:")
                for algo in algorithms:
                    lines.append(f"  - {algo}")

        return "\n".join(lines)


# Create global registry instance
algorithm_registry = AlgorithmRegistry()
