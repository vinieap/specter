"""
Comprehensive tests for the clustering framework API.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from clustering_framework import (
    optimize_clustering,
    analyze_clusters,
    evaluate_clustering,
    quick_cluster,
)


class TestOptimizeClustering:
    """Test clustering optimization functionality."""

    def test_kmeans_optimization(
        self, blobs_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test K-means optimization on well-separated data."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=20, random_state=42, n_clusters=4
        )

        assert results.best_score > 0.7  # Good separation should yield high silhouette
        assert isinstance(results.best_model, BaseEstimator)
        assert len(results.history) == 20
        assert all(h["score"] > 0 for h in results.history)

    def test_dbscan_optimization(
        self, moons_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test DBSCAN optimization on non-spherical clusters."""
        X, _ = moons_data

        results = optimize_clustering(
            X, algorithm="dbscan", n_calls=20, eps=0.3, min_samples=5
        )

        assert (
            results.best_score > 0.3
        )  # Non-spherical clusters might have lower scores
        assert len(results.history) == 20

    def test_spectral_optimization(
        self, circles_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test Spectral clustering on concentric circles."""
        X, _ = circles_data

        results = optimize_clustering(
            X,
            algorithm="spectral",
            n_calls=20,
            random_state=42,
            n_clusters=2,
            affinity="nearest_neighbors",
        )

        assert results.best_score > 0.3  # Concentric circles are challenging
        assert len(results.history) == 20

    def test_invalid_algorithm(self, blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test error handling for invalid algorithm."""
        X, _ = blobs_data

        with pytest.raises(ValueError, match="Unknown algorithm"):
            optimize_clustering(X, algorithm="invalid_algo")


class TestAnalyzeClusters:
    """Test cluster analysis functionality."""

    def test_stability_analysis(
        self, blobs_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test stability analysis on well-separated data."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=10, random_state=42, n_clusters=4
        )

        analysis = analyze_clusters(
            X, results.best_model, stability_analysis=True, noise_analysis=False
        )

        assert "stability_scores" in analysis
        assert analysis["stability_scores"]["mean_stability"] > 0.7

    def test_noise_analysis(self, blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test noise sensitivity analysis."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=10, random_state=42, n_clusters=4
        )

        analysis = analyze_clusters(
            X, results.best_model, stability_analysis=False, noise_analysis=True
        )

        assert "noise_analysis" in analysis
        assert all(
            analysis["noise_analysis"][f"noise_{level}"]["mean_score"] > 0.5
            for level in [0.01, 0.05, 0.1]
        )

    def test_full_analysis(self, blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test both stability and noise analysis."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=10, random_state=42, n_clusters=4
        )

        analysis = analyze_clusters(
            X, results.best_model, stability_analysis=True, noise_analysis=True
        )

        assert "stability_scores" in analysis
        assert "noise_analysis" in analysis


class TestEvaluateClustering:
    """Test clustering evaluation metrics."""

    @pytest.mark.parametrize(
        "metric", ["silhouette", "calinski_harabasz", "davies_bouldin"]
    )
    def test_individual_metrics(
        self, blobs_data: Tuple[np.ndarray, np.ndarray], metric: str
    ) -> None:
        """Test each evaluation metric individually."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=10, random_state=42, n_clusters=4
        )

        metrics = evaluate_clustering(X, results.best_model, metrics=[metric])
        assert metric in metrics
        assert metrics[metric] > 0

    def test_all_metrics(self, blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing all available metrics."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=10, random_state=42, n_clusters=4
        )

        metrics = evaluate_clustering(
            X,
            results.best_model,
            metrics=["silhouette", "calinski_harabasz", "davies_bouldin"],
        )

        assert len(metrics) == 3
        assert all(score > 0 for score in metrics.values())

    def test_invalid_metric(self, blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test error handling for invalid metric."""
        X, _ = blobs_data

        results = optimize_clustering(
            X, algorithm="kmeans", n_calls=10, random_state=42, n_clusters=4
        )

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_clustering(X, results.best_model, metrics=["invalid_metric"])


class TestQuickCluster:
    """Test quick clustering functionality."""

    @pytest.mark.parametrize(
        "algorithm,params",
        [
            ("kmeans", {"n_clusters": 4}),
            ("dbscan", {"eps": 0.3, "min_samples": 5}),
            ("spectral", {"n_clusters": 2, "affinity": "nearest_neighbors"}),
        ],
    )
    def test_different_algorithms(
        self,
        blobs_data: Tuple[np.ndarray, np.ndarray],
        algorithm: str,
        params: Dict[str, Any],
    ) -> None:
        """Test quick clustering with different algorithms."""
        X, _ = blobs_data

        model, metrics = quick_cluster(
            X, algorithm=algorithm, random_state=42, **params
        )

        assert isinstance(model, BaseEstimator)
        assert "silhouette" in metrics
        assert metrics["silhouette"] > 0

    def test_custom_parameters(self, blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test quick clustering with custom parameters."""
        X, _ = blobs_data

        model, metrics = quick_cluster(
            X,
            n_clusters=4,
            algorithm="kmeans",
            random_state=42,
            init="random",
            max_iter=500,
        )

        assert isinstance(model, BaseEstimator)
        assert metrics["silhouette"] > 0

    def test_high_dimensional_data(self) -> None:
        """Test clustering on high-dimensional data."""
        X, _ = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
        X = StandardScaler().fit_transform(X)

        model, metrics = quick_cluster(
            X, n_clusters=3, algorithm="kmeans", random_state=42
        )

        assert isinstance(model, BaseEstimator)
        assert metrics["silhouette"] > 0


def test_end_to_end_workflow(blobs_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test complete clustering workflow."""
    X, _ = blobs_data

    # 1. Optimize clustering
    opt_results = optimize_clustering(
        X, algorithm="kmeans", n_calls=20, random_state=42, n_clusters=4
    )
    assert opt_results.best_score > 0

    # 2. Analyze clusters
    analysis = analyze_clusters(
        X, opt_results.best_model, stability_analysis=True, noise_analysis=True
    )
    assert "stability_scores" in analysis
    assert "noise_analysis" in analysis

    # 3. Evaluate with multiple metrics
    metrics = evaluate_clustering(
        X,
        opt_results.best_model,
        metrics=["silhouette", "calinski_harabasz", "davies_bouldin"],
    )
    assert len(metrics) == 3
    assert all(score > 0 for score in metrics.values())
