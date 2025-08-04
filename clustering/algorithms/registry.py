"""Registry for clustering algorithms."""
from typing import Dict, Type

from .base import BaseClusteringAlgorithm
from .spectral import SpectralClusteringAlgorithm
from .kmeans import KMeansClusteringAlgorithm
from .dbscan import DBSCANClusteringAlgorithm
from .hdbscan import HDBSCANClusteringAlgorithm
from .affinity_propagation import AffinityPropagationClusteringAlgorithm
from .agglomerative import AgglomerativeClusteringAlgorithm
from .birch import BirchClusteringAlgorithm
from .bisecting_kmeans import BisectingKMeansClusteringAlgorithm
from .mean_shift import MeanShiftClusteringAlgorithm
from .mini_batch_kmeans import MiniBatchKMeansClusteringAlgorithm
from .optics import OPTICSClusteringAlgorithm


class ClusteringAlgorithmRegistry:
    """Registry for managing available clustering algorithms."""

    def __init__(self):
        self._algorithms: Dict[str, Type[BaseClusteringAlgorithm]] = {}
        self._register_default_algorithms()

    def _register_default_algorithms(self):
        """Register the default set of clustering algorithms."""
        # Basic clustering algorithms
        self.register_algorithm(KMeansClusteringAlgorithm)
        self.register_algorithm(MiniBatchKMeansClusteringAlgorithm)
        self.register_algorithm(BisectingKMeansClusteringAlgorithm)

        # Density-based clustering
        self.register_algorithm(DBSCANClusteringAlgorithm)
        self.register_algorithm(HDBSCANClusteringAlgorithm)
        self.register_algorithm(OPTICSClusteringAlgorithm)

        # Hierarchical clustering
        self.register_algorithm(AgglomerativeClusteringAlgorithm)
        self.register_algorithm(BirchClusteringAlgorithm)

        # Affinity/similarity-based clustering
        self.register_algorithm(AffinityPropagationClusteringAlgorithm)
        self.register_algorithm(SpectralClusteringAlgorithm)
        self.register_algorithm(MeanShiftClusteringAlgorithm)

    def register_algorithm(self, algorithm_class: Type[BaseClusteringAlgorithm]):
        """Register a new clustering algorithm.

        Parameters
        ----------
        algorithm_class : Type[BaseClusteringAlgorithm]
            The clustering algorithm class to register
        """
        # Create a temporary instance to get the name
        temp_instance = algorithm_class()
        self._algorithms[temp_instance.name] = algorithm_class

    def get_algorithm(self, name: str, random_state: int = 42) -> BaseClusteringAlgorithm:
        """Get an instance of a clustering algorithm by name.

        Parameters
        ----------
        name : str
            Name of the algorithm to get
        random_state : int, default=42
            Random state for reproducibility

        Returns
        -------
        BaseClusteringAlgorithm
            Instance of the requested algorithm

        Raises
        ------
        ValueError
            If the algorithm name is not registered
        """
        algorithm_class = self._algorithms.get(name.lower())
        if algorithm_class is None:
            available = list(self._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm: {name}. Available algorithms: {available}"
            )
        return algorithm_class(random_state=random_state)

    def list_algorithms(self) -> list:
        """Get a list of all registered algorithm names.

        Returns
        -------
        list
            List of algorithm names
        """
        return sorted(self._algorithms.keys())

    def get_algorithm_categories(self) -> Dict[str, list]:
        """Get algorithms grouped by category.

        Returns
        -------
        Dict[str, list]
            Dictionary mapping category names to lists of algorithm names
        """
        categories = {
            "Basic Clustering": ["kmeans", "mini_batch_kmeans", "bisecting_kmeans"],
            "Density-Based": ["dbscan", "hdbscan", "optics"],
            "Hierarchical": ["agglomerative", "birch"],
            "Affinity/Similarity-Based": ["affinity_propagation", "spectral", "mean_shift"],
        }
        return categories


# Create a global instance of the registry
algorithm_registry = ClusteringAlgorithmRegistry()