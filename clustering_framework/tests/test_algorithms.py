"""
Tests for all clustering algorithms and their optimization.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler

from clustering_framework.algorithms import (
    ClusteringAlgorithm,
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
from clustering_framework.optimizers.batch import BatchOptimizer
from clustering_framework.optimizers.sequential import SequentialOptimizer


@pytest.fixture
def sparse_blobs():
    """Generate sparse, well-separated clusters."""
    X, y = make_blobs(
        n_samples=300, n_features=2, centers=5, cluster_std=0.5, random_state=42
    )
    return StandardScaler().fit_transform(X), y


@pytest.fixture
def dense_blobs():
    """Generate dense, overlapping clusters."""
    X, y = make_blobs(
        n_samples=300, n_features=2, centers=3, cluster_std=2.0, random_state=42
    )
    return StandardScaler().fit_transform(X), y


@pytest.fixture
def high_dim_data():
    """Generate high-dimensional data with clear cluster structure."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        n_classes=3,
        random_state=42,
    )
    return StandardScaler().fit_transform(X), y


@pytest.fixture
def noisy_data():
    """Generate data with significant noise."""
    X, y = make_blobs(
        n_samples=300, n_features=2, centers=4, cluster_std=1.5, random_state=42
    )
    # Add random noise
    noise = np.random.RandomState(42).normal(0, 0.5, X.shape)
    X = X + noise
    return StandardScaler().fit_transform(X), y


def test_algorithm_registry():
    """Test that all algorithms are properly registered."""
    from clustering_framework.core.registry import AlgorithmRegistry

    # Create a fresh registry without default algorithms
    registry = AlgorithmRegistry(register_defaults=False)

    algorithms = [
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
    ]

    # Check that each algorithm has the required properties
    for algo_class in algorithms:
        # Register the algorithm
        registry.register_algorithm(algo_class)

        # Basic property checks
        algo = algo_class()
        assert isinstance(algo, ClusteringAlgorithm)
        assert algo.name is not None
        assert algo.category in ClusteringAlgorithm.CATEGORIES
        assert hasattr(algo, "estimator_class")
        assert hasattr(algo, "get_default_parameters")
        assert hasattr(algo, "sample_parameters")
        assert hasattr(algo, "validate_parameters")

        # Test name caching
        cached_name = registry._get_algorithm_name(algo_class)
        assert hasattr(algo_class, "_algorithm_name")
        assert algo_class._algorithm_name.lower() == cached_name

        # Test category caching
        cached_category = registry._get_algorithm_category(algo_class)
        assert hasattr(algo_class, "_algorithm_category")
        assert algo_class._algorithm_category == cached_category

        # Test instance caching
        instance1 = registry.get_algorithm(algo.name, random_state=42)
        instance2 = registry.get_algorithm(algo.name, random_state=42)
        assert instance1 is instance2  # Same instance should be returned

        # Test cache cleanup
        for i in range(registry._max_cache_size + 5):
            registry.get_algorithm(algo.name, random_state=i)
        assert len(registry._instance_cache[algo.name]) <= registry._max_cache_size

        # Test different random states get different instances
        instance3 = registry.get_algorithm(algo.name, random_state=43)
        assert instance1 is not instance3


@pytest.mark.parametrize("optimizer_class", [BatchOptimizer, SequentialOptimizer])
class TestAlgorithmOptimization:
    """Test optimization for each algorithm type."""

    def test_basic_clustering(self, sparse_blobs, optimizer_class):
        """Test basic clustering algorithms (K-means variants)."""
        X, _ = sparse_blobs

        for algo in [
            KMeansAlgorithm,
            MiniBatchKMeansAlgorithm,
            BisectingKMeansAlgorithm,
        ]:
            optimizer = optimizer_class(
                algorithm=algo(), max_trials=20, random_state=42
            )

            results = optimizer.optimize(X)
            assert results.best_score > 0.5  # Good separation should yield high scores
            assert results.best_model is not None

            # Test prediction if supported
            if hasattr(results.best_model, "labels_"):
                labels = results.best_model.labels_
            elif hasattr(results.best_model, "predict"):
                labels = results.best_model.predict(X)
            else:
                raise ValueError(f"{algo.__name__} does not provide cluster labels")
            assert len(np.unique(labels)) > 1  # Should find multiple clusters

    def test_density_based(self, dense_blobs, optimizer_class):
        """Test density-based clustering algorithms."""
        X, _ = dense_blobs

        for algo in [DBSCANAlgorithm, HDBSCANAlgorithm, OPTICSAlgorithm]:
            optimizer = optimizer_class(
                algorithm=algo(), max_trials=20, random_state=42
            )

            results = optimizer.optimize(X)
            assert (
                results.best_score > -0.2
            )  # Scores might be negative due to noise handling
            assert results.best_model is not None

            # Test prediction if supported
            if hasattr(results.best_model, "labels_"):
                labels = results.best_model.labels_
            elif hasattr(results.best_model, "predict"):
                labels = results.best_model.predict(X)
            else:
                raise ValueError(f"{algo.__name__} does not provide cluster labels")
            assert len(np.unique(labels)) > 1  # Should find multiple clusters

    def test_hierarchical(self, sparse_blobs, optimizer_class):
        """Test hierarchical clustering algorithms."""
        X, _ = sparse_blobs

        for algo in [AgglomerativeClusteringAlgorithm, BirchAlgorithm]:
            optimizer = optimizer_class(
                algorithm=algo(), max_trials=20, random_state=42
            )

            results = optimizer.optimize(X)
            assert (
                results.best_score > -0.2
            )  # Scores might be negative for some linkage types
            assert results.best_model is not None

            # Test prediction if supported
            if hasattr(results.best_model, "labels_"):
                labels = results.best_model.labels_
            elif hasattr(results.best_model, "predict"):
                labels = results.best_model.predict(X)
            else:
                raise ValueError(f"{algo.__name__} does not provide cluster labels")
            assert len(np.unique(labels)) > 1  # Should find multiple clusters

    def test_affinity_based(self, sparse_blobs, optimizer_class):
        """Test affinity/similarity-based clustering algorithms."""
        X, _ = sparse_blobs

        for algo in [
            AffinityPropagationAlgorithm,
            SpectralClusteringAlgorithm,
            MeanShiftAlgorithm,
        ]:
            optimizer = optimizer_class(
                algorithm=algo(), max_trials=20, random_state=42
            )

            results = optimizer.optimize(X)
            assert (
                results.best_score > -0.2
            )  # Scores might be negative for some affinity types
            assert results.best_model is not None

            # Test prediction if supported
            if hasattr(results.best_model, "labels_"):
                labels = results.best_model.labels_
            elif hasattr(results.best_model, "predict"):
                labels = results.best_model.predict(X)
            else:
                raise ValueError(f"{algo.__name__} does not provide cluster labels")
            assert len(np.unique(labels)) > 1  # Should find multiple clusters


class TestSpecializedCases:
    """Test algorithms on specific data characteristics."""

    def test_high_dimensional(self, high_dim_data):
        """Test performance on high-dimensional data."""
        X, _ = high_dim_data

        # These algorithms typically handle high-dimensional data well
        good_for_high_dim = [KMeansAlgorithm, MiniBatchKMeansAlgorithm, BirchAlgorithm]

        for algo in good_for_high_dim:
            optimizer = BatchOptimizer(algorithm=algo(), max_trials=20, random_state=42)

            results = optimizer.optimize(X)
            assert (
                results.best_score > 0.05
            )  # Even low scores are acceptable for high-dim data

    def test_noisy_data(self, noisy_data):
        """Test performance on noisy data."""
        X, _ = noisy_data

        # These algorithms are typically robust to noise
        noise_robust = [HDBSCANAlgorithm, DBSCANAlgorithm, SpectralClusteringAlgorithm]

        for algo in noise_robust:
            optimizer = BatchOptimizer(algorithm=algo(), max_trials=20, random_state=42)

            results = optimizer.optimize(X)
            assert results.best_score > 0  # Scores will be lower due to noise

    def test_parameter_spaces(self):
        """Test that each algorithm defines valid parameter spaces."""
        algorithms = [
            KMeansAlgorithm(),
            MiniBatchKMeansAlgorithm(),
            BisectingKMeansAlgorithm(),
            DBSCANAlgorithm(),
            HDBSCANAlgorithm(),
            OPTICSAlgorithm(),
            AgglomerativeClusteringAlgorithm(),
            BirchAlgorithm(),
            AffinityPropagationAlgorithm(),
            SpectralClusteringAlgorithm(),
            MeanShiftAlgorithm(),
        ]

        for algo in algorithms:
            params = algo.get_default_parameters()

            assert isinstance(params, dict)
            algo.validate_parameters(params)  # Should not raise any errors


def test_optuna_integration():
    """Test proper integration with Optuna's optimization."""
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    X = StandardScaler().fit_transform(X)

    algorithms = [
        KMeansAlgorithm(),
        MiniBatchKMeansAlgorithm(),
        BisectingKMeansAlgorithm(),
        DBSCANAlgorithm(),
        HDBSCANAlgorithm(),
        OPTICSAlgorithm(),
        AgglomerativeClusteringAlgorithm(),
        BirchAlgorithm(),
        AffinityPropagationAlgorithm(),
        SpectralClusteringAlgorithm(),
        MeanShiftAlgorithm(),
    ]

    for algo in algorithms:
        optimizer = BatchOptimizer(
            algorithm=algo,
            max_trials=10,
            random_state=42,
            use_dashboard=True,  # Test dashboard integration
        )

        results = optimizer.optimize(X)

        # Check Optuna study properties
        assert results.study is not None
        assert len(results.study.trials) == 10
        assert results.study.best_trial is not None

        # Check that parameters match
        assert results.best_params == results.study.best_trial.params
        assert results.best_score == results.study.best_trial.value


def test_parallel_optimization():
    """Test parallel optimization capabilities."""
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Test with different batch sizes
    batch_sizes = [2, 4, 8]
    for batch_size in batch_sizes:
        optimizer = BatchOptimizer(
            algorithm=KMeansAlgorithm(),
            max_trials=10,
            batch_size=batch_size,
            random_state=42,
        )

        results = optimizer.optimize(X)

        # Check that batches were processed
        assert (
            results.execution_stats["n_batches"] == (10 + batch_size - 1) // batch_size
        )
        assert results.execution_stats["batch_size"] == batch_size
