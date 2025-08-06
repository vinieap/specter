"""Quick clustering functionality."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from .registry import algorithm_registry
from .metrics import ClusteringMetrics
from .analysis import ClusteringAnalyzer


class QuickCluster:
    """Quick clustering interface.

    This class provides a simple interface for quick clustering without
    going through the full optimization process.
    """

    # Default parameters for each algorithm
    DEFAULT_PARAMS = {
        "kmeans": {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
        },
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
        },
        "spectral": {
            "n_clusters": 8,
            "affinity": "rbf",
            "n_neighbors": 10,
        },
        "agglomerative": {
            "n_clusters": 8,
            "affinity": "euclidean",
            "linkage": "ward",
        },
        "birch": {
            "n_clusters": 8,
            "threshold": 0.5,
            "branching_factor": 50,
        },
        "optics": {
            "min_samples": 5,
            "max_eps": 0.5,
            "metric": "minkowski",
        },
        "mean_shift": {
            "bandwidth": None,
            "bin_seeding": False,
            "cluster_all": True,
        },
        "affinity_propagation": {
            "damping": 0.5,
            "max_iter": 200,
            "convergence_iter": 15,
        },
    }

    def __init__(
        self,
        algorithm: str = "kmeans",
        random_state: Optional[int] = None,
        scale_data: bool = True,
        analyze_results: bool = True,
        **kwargs: Any,
    ):
        """Initialize quick clustering.

        Args:
            algorithm: Name of the clustering algorithm
            random_state: Random seed for reproducibility
            scale_data: Whether to scale input data
            analyze_results: Whether to analyze clustering results
            **kwargs: Additional parameters for the algorithm
        """
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        self.scale_data = scale_data
        self.analyze_results = analyze_results
        self.kwargs = kwargs

        # Get algorithm instance
        self.model = algorithm_registry.get_algorithm(
            algorithm, random_state=random_state
        )

        # Set parameters
        params = self.DEFAULT_PARAMS.get(self.algorithm, {}).copy()
        params.update(kwargs)
        self.model.set_parameters(params)

        # Initialize components
        self.scaler = StandardScaler() if scale_data else None
        self.analyzer = ClusteringAnalyzer(
            random_state=random_state
        ) if analyze_results else None

    def fit(
        self, X: np.ndarray
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Fit clustering model.

        Args:
            X: Input data matrix

        Returns:
            Tuple of (fitted model, results dictionary)
        """
        # Scale data if requested
        if self.scale_data:
            X = self.scaler.fit_transform(X)

        # Fit model
        model = self.model.fit(X)

        # Get labels
        labels = model.predict(X) if hasattr(model, "predict") else model.labels_

        # Compute basic metrics
        results = {
            "metrics": ClusteringMetrics.compute_metrics(
                labels,
                metrics=["silhouette", "calinski_harabasz", "davies_bouldin"],
                X=X,
            ),
            "n_clusters": len(np.unique(labels[labels != -1])),
            "cluster_sizes": ClusteringMetrics.compute_cluster_sizes(labels),
        }

        # Analyze results if requested
        if self.analyze_results:
            results["stability"] = self.analyzer.analyze_stability(
                X, model
            )
            results["noise"] = self.analyzer.analyze_noise(X, model)
            results["characteristics"] = (
                self.analyzer.analyze_cluster_characteristics(X, model)
            )

        return model, results

    def fit_predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit model and predict cluster labels.

        Args:
            X: Input data matrix

        Returns:
            Tuple of (cluster labels, results dictionary)
        """
        model, results = self.fit(X)
        labels = model.predict(X) if hasattr(model, "predict") else model.labels_
        return labels, results

    @classmethod
    def get_default_params(cls, algorithm: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm.

        Args:
            algorithm: Name of the algorithm

        Returns:
            Dictionary of default parameters

        Raises:
            ValueError: If algorithm is not recognized
        """
        if algorithm not in cls.DEFAULT_PARAMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return cls.DEFAULT_PARAMS[algorithm].copy()

    @classmethod
    def list_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """List available algorithms and their default parameters.

        Returns:
            Dictionary mapping algorithm names to default parameters
        """
        return {
            name: {
                "params": params.copy(),
                "description": algorithm_registry.get_algorithm(name).name,
            }
            for name, params in cls.DEFAULT_PARAMS.items()
        }