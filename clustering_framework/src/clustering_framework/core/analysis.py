"""Stability and noise analysis for clustering results."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .metrics import ClusteringMetrics


@dataclass
class StabilityAnalysis:
    """Results of stability analysis.

    Attributes:
        mean_stability: Mean stability score across perturbations
        std_stability: Standard deviation of stability scores
        stability_scores: List of individual stability scores
        label_consistency: Consistency of label assignments
        cluster_persistence: Persistence of cluster structures
    """

    mean_stability: float
    std_stability: float
    stability_scores: List[float]
    label_consistency: Dict[int, float]
    cluster_persistence: Dict[int, float]


@dataclass
class NoiseAnalysis:
    """Results of noise analysis.

    Attributes:
        noise_ratio: Ratio of points identified as noise
        noise_indices: Indices of noise points
        noise_scores: Noise scores for each point
        threshold: Noise detection threshold
        local_density: Local density estimates
        outlier_scores: Outlier scores
    """

    noise_ratio: float
    noise_indices: np.ndarray
    noise_scores: np.ndarray
    threshold: float
    local_density: np.ndarray
    outlier_scores: np.ndarray


class ClusteringAnalyzer:
    """Analyzer for clustering results.

    This class provides methods for analyzing the stability and noise
    characteristics of clustering results.
    """

    def __init__(
        self,
        n_perturbations: int = 10,
        noise_threshold: float = 2.0,
        random_state: Optional[int] = None,
    ):
        """Initialize analyzer.

        Args:
            n_perturbations: Number of perturbations for stability analysis
            noise_threshold: Threshold for noise detection (in std deviations)
            random_state: Random seed for reproducibility
        """
        self.n_perturbations = n_perturbations
        self.noise_threshold = noise_threshold
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def analyze_stability(
        self,
        X: np.ndarray,
        model: BaseEstimator,
        noise_scale: float = 0.1,
        subsample_ratio: float = 0.9,
    ) -> StabilityAnalysis:
        """Analyze clustering stability.

        Args:
            X: Input data matrix
            model: Fitted clustering model
            noise_scale: Scale of random perturbations
            subsample_ratio: Ratio of data to use in each perturbation

        Returns:
            Stability analysis results
        """
        n_samples = X.shape[0]
        subsample_size = int(n_samples * subsample_ratio)
        base_labels = model.predict(X) if hasattr(model, "predict") else model.labels_

        stability_scores = []
        label_counts = {}
        cluster_counts = {}

        for _ in range(self.n_perturbations):
            # Add noise and subsample
            noise = self.rng.normal(0, noise_scale, X.shape)
            indices = self.rng.choice(
                n_samples, subsample_size, replace=False
            )
            X_perturbed = X[indices] + noise[indices]

            # Get labels for perturbed data
            if hasattr(model, "predict"):
                labels = model.predict(X_perturbed)
            else:
                # Refit for algorithms without predict
                model.fit(X_perturbed)
                labels = model.labels_

            # Compute stability metrics
            stability = ClusteringMetrics.compute_cluster_stability(
                base_labels[indices], labels
            )
            stability_scores.append(stability)

            # Track label consistency
            for i, label in enumerate(labels):
                orig_label = base_labels[indices[i]]
                if orig_label not in label_counts:
                    label_counts[orig_label] = {}
                if label not in label_counts[orig_label]:
                    label_counts[orig_label][label] = 0
                label_counts[orig_label][label] += 1

            # Track cluster persistence
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label not in cluster_counts:
                    cluster_counts[label] = 0
                cluster_counts[label] += 1

        # Compute label consistency
        label_consistency = {}
        for orig_label, counts in label_counts.items():
            total = sum(counts.values())
            max_count = max(counts.values())
            label_consistency[orig_label] = max_count / total

        # Compute cluster persistence
        cluster_persistence = {}
        for label, count in cluster_counts.items():
            cluster_persistence[label] = count / self.n_perturbations

        return StabilityAnalysis(
            mean_stability=np.mean(stability_scores),
            std_stability=np.std(stability_scores),
            stability_scores=stability_scores,
            label_consistency=label_consistency,
            cluster_persistence=cluster_persistence,
        )

    def analyze_noise(
        self,
        X: np.ndarray,
        model: BaseEstimator,
        k_neighbors: int = 5,
    ) -> NoiseAnalysis:
        """Analyze noise and outliers in clustering.

        Args:
            X: Input data matrix
            model: Fitted clustering model
            k_neighbors: Number of neighbors for density estimation

        Returns:
            Noise analysis results
        """
        # Compute local density
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)

        # Compute cluster-based outlier scores
        labels = model.predict(X) if hasattr(model, "predict") else model.labels_
        outlier_scores = np.zeros(X.shape[0])

        for label in np.unique(labels):
            if label == -1:  # Skip noise cluster
                continue
            mask = labels == label
            cluster_points = X[mask]
            if len(cluster_points) > 1:
                # Compute distance to cluster centroid
                centroid = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(
                    cluster_points - centroid, axis=1
                )
                outlier_scores[mask] = distances

        # Normalize scores
        outlier_scores = (outlier_scores - np.mean(outlier_scores)) / (
            np.std(outlier_scores) + 1e-10
        )

        # Combine density and outlier scores
        noise_scores = outlier_scores - np.log1p(local_density)
        noise_scores = (noise_scores - np.mean(noise_scores)) / (
            np.std(noise_scores) + 1e-10
        )

        # Detect noise points
        noise_indices = np.where(noise_scores > self.noise_threshold)[0]
        noise_ratio = len(noise_indices) / len(X)

        return NoiseAnalysis(
            noise_ratio=noise_ratio,
            noise_indices=noise_indices,
            noise_scores=noise_scores,
            threshold=self.noise_threshold,
            local_density=local_density,
            outlier_scores=outlier_scores,
        )

    def analyze_cluster_characteristics(
        self,
        X: np.ndarray,
        model: BaseEstimator,
    ) -> Dict[str, Any]:
        """Analyze characteristics of each cluster.

        Args:
            X: Input data matrix
            model: Fitted clustering model

        Returns:
            Dictionary containing cluster characteristics
        """
        labels = model.predict(X) if hasattr(model, "predict") else model.labels_
        unique_labels = np.unique(labels)

        characteristics = {
            "sizes": ClusteringMetrics.compute_cluster_sizes(labels),
            "densities": ClusteringMetrics.compute_cluster_density(X, labels),
            "separation": ClusteringMetrics.compute_cluster_separation(X, labels),
        }

        # Compute within-cluster statistics
        within_stats = {}
        for label in unique_labels:
            if label == -1:  # Skip noise cluster
                continue
            mask = labels == label
            points = X[mask]
            if len(points) > 1:
                within_stats[label] = {
                    "mean": np.mean(points, axis=0),
                    "std": np.std(points, axis=0),
                    "radius": np.max(
                        np.linalg.norm(
                            points - np.mean(points, axis=0), axis=1
                        )
                    ),
                }

        characteristics["within_stats"] = within_stats

        return characteristics