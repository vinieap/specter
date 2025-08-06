"""Clustering evaluation metrics."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix


class ClusteringMetrics:
    """Collection of clustering evaluation metrics.

    This class provides a unified interface for computing various
    clustering evaluation metrics, including internal and external
    validation measures.
    """

    # Available metrics with their functions and descriptions
    AVAILABLE_METRICS = {
        "silhouette": {
            "func": silhouette_score,
            "desc": "Silhouette coefficient (-1 to 1, higher is better)",
            "requires_features": True,
        },
        "calinski_harabasz": {
            "func": calinski_harabasz_score,
            "desc": "Calinski-Harabasz index (higher is better)",
            "requires_features": True,
        },
        "davies_bouldin": {
            "func": davies_bouldin_score,
            "desc": "Davies-Bouldin index (lower is better)",
            "requires_features": True,
        },
        "adjusted_rand": {
            "func": adjusted_rand_score,
            "desc": "Adjusted Rand index (-1 to 1, higher is better)",
            "requires_features": False,
        },
        "adjusted_mutual_info": {
            "func": adjusted_mutual_info_score,
            "desc": "Adjusted mutual information (0 to 1, higher is better)",
            "requires_features": False,
        },
        "v_measure": {
            "func": v_measure_score,
            "desc": "V-measure (0 to 1, higher is better)",
            "requires_features": False,
        },
    }

    @classmethod
    def get_available_metrics(cls) -> Dict[str, str]:
        """Get available metrics and their descriptions.

        Returns:
            Dictionary mapping metric names to descriptions
        """
        return {
            name: info["desc"]
            for name, info in cls.AVAILABLE_METRICS.items()
        }

    @classmethod
    def requires_features(cls, metric: str) -> bool:
        """Check if a metric requires feature data.

        Args:
            metric: Name of the metric

        Returns:
            True if metric requires feature data, False otherwise

        Raises:
            ValueError: If metric is not recognized
        """
        if metric not in cls.AVAILABLE_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        return cls.AVAILABLE_METRICS[metric]["requires_features"]

    @classmethod
    def compute_metric(
        cls,
        metric: str,
        labels: np.ndarray,
        X: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> float:
        """Compute a single clustering metric.

        Args:
            metric: Name of the metric to compute
            labels: Predicted cluster labels
            X: Input feature matrix (required for some metrics)
            true_labels: True cluster labels (required for some metrics)
            **kwargs: Additional parameters for the metric

        Returns:
            Metric value

        Raises:
            ValueError: If metric is not recognized or required data is missing
        """
        if metric not in cls.AVAILABLE_METRICS:
            raise ValueError(f"Unknown metric: {metric}")

        metric_info = cls.AVAILABLE_METRICS[metric]

        if metric_info["requires_features"] and X is None:
            raise ValueError(f"Metric {metric} requires feature data (X)")

        if metric in ["adjusted_rand", "adjusted_mutual_info", "v_measure"]:
            if true_labels is None:
                raise ValueError(f"Metric {metric} requires true labels")
            return metric_info["func"](true_labels, labels, **kwargs)

        return metric_info["func"](X, labels, **kwargs)

    @classmethod
    def compute_metrics(
        cls,
        labels: np.ndarray,
        metrics: Optional[List[str]] = None,
        X: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute multiple clustering metrics.

        Args:
            labels: Predicted cluster labels
            metrics: List of metrics to compute (default: all available)
            X: Input feature matrix (required for some metrics)
            true_labels: True cluster labels (required for some metrics)
            **kwargs: Additional parameters for metrics

        Returns:
            Dictionary mapping metric names to values

        Raises:
            ValueError: If any metric is not recognized or required data is missing
        """
        if metrics is None:
            metrics = list(cls.AVAILABLE_METRICS.keys())

        results = {}
        for metric in metrics:
            try:
                results[metric] = cls.compute_metric(
                    metric, labels, X, true_labels, **kwargs
                )
            except ValueError as e:
                # Skip metrics that can't be computed
                continue

        return results

    @classmethod
    def compute_cluster_stability(
        cls,
        labels1: np.ndarray,
        labels2: np.ndarray,
        method: str = "adjusted_rand",
    ) -> float:
        """Compute stability between two clustering results.

        Args:
            labels1: First set of cluster labels
            labels2: Second set of cluster labels
            method: Stability metric to use

        Returns:
            Stability score

        Raises:
            ValueError: If method is not recognized
        """
        if method == "adjusted_rand":
            return adjusted_rand_score(labels1, labels2)
        elif method == "adjusted_mutual_info":
            return adjusted_mutual_info_score(labels1, labels2)
        elif method == "v_measure":
            return v_measure_score(labels1, labels2)
        else:
            raise ValueError(f"Unknown stability method: {method}")

    @classmethod
    def compute_cluster_sizes(cls, labels: np.ndarray) -> Dict[int, int]:
        """Compute sizes of each cluster.

        Args:
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster labels to sizes
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))

    @classmethod
    def compute_cluster_density(
        cls, X: np.ndarray, labels: np.ndarray
    ) -> Dict[int, float]:
        """Compute density of each cluster.

        Args:
            X: Input feature matrix
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster labels to densities
        """
        densities = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            points = X[mask]
            if len(points) > 1:
                # Compute average pairwise distance within cluster
                dists = np.linalg.norm(
                    points[:, np.newaxis] - points, axis=2
                )
                density = 1.0 / (np.mean(dists) + 1e-10)
                densities[label] = density
            else:
                densities[label] = 0.0

        return densities

    @classmethod
    def compute_cluster_separation(
        cls, X: np.ndarray, labels: np.ndarray
    ) -> Dict[int, float]:
        """Compute separation between clusters.

        Args:
            X: Input feature matrix
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster pairs to separation scores
        """
        unique_labels = np.unique(labels)
        centroids = {}
        separation = {}

        # Compute centroids
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            centroids[label] = np.mean(X[mask], axis=0)

        # Compute pairwise distances between centroids
        for i in unique_labels:
            if i == -1:
                continue
            for j in unique_labels:
                if j == -1 or j <= i:
                    continue
                dist = np.linalg.norm(centroids[i] - centroids[j])
                separation[(i, j)] = dist

        return separation