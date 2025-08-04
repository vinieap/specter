"""Registry system for clustering algorithms."""
from typing import Dict, Type, List, Optional

from .algorithm import ClusteringAlgorithm


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
    
    def __init__(self):
        """Initialize the registry."""
        self._algorithms: Dict[str, Type[ClusteringAlgorithm]] = {}
    
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
        
        # Create temporary instance to get name
        temp_instance = algorithm_class()
        name = temp_instance.name.lower()
        
        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' is already registered")
        
        self._algorithms[name] = algorithm_class
    
    def get_algorithm(
        self,
        name: str,
        random_state: Optional[int] = None
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
        algorithm_class = self._algorithms.get(name.lower())
        if algorithm_class is None:
            available = list(self._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm: {name}. "
                f"Available algorithms: {available}"
            )
        return algorithm_class(random_state=random_state)
    
    def list_algorithms(self) -> List[str]:
        """Get a list of all registered algorithm names.
        
        Returns
        -------
        List[str]
            List of algorithm names
        """
        return sorted(self._algorithms.keys())
    
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
            # Create temporary instance to get category
            temp_instance = algo_class()
            category = ClusteringAlgorithm.CATEGORIES[temp_instance.category]
            categories[category].append(name)
        
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