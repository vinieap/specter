"""Tests for enhanced convergence detection."""

import numpy as np
import pytest
from sklearn.cluster import KMeans

from clustering_framework.core.convergence import (
    ConvergenceCriterion,
    ConvergenceConfig,
    ConvergenceState,
    ConvergenceDetector,
)


def test_convergence_config_validation():
    """Test validation of convergence configuration."""
    # Test valid configuration
    config = ConvergenceConfig(
        criteria=[ConvergenceCriterion.INERTIA],
        thresholds={ConvergenceCriterion.INERTIA: 1e-4},
    )
    detector = ConvergenceDetector(config)
    assert detector.config == config

    # Test missing criteria
    with pytest.raises(ValueError, match="At least one convergence criterion"):
        detector = ConvergenceDetector(ConvergenceConfig(criteria=[], thresholds={}))

    # Test invalid criteria type
    with pytest.raises(ValueError, match="criteria must be a list"):
        detector = ConvergenceDetector(ConvergenceConfig(criteria=None, thresholds={}))

    # Test invalid criterion type
    with pytest.raises(ValueError, match="Invalid criterion type"):
        detector = ConvergenceDetector(ConvergenceConfig(criteria=["invalid"], thresholds={}))

    # Test missing threshold
    with pytest.raises(ValueError, match="Threshold not specified"):
        detector = ConvergenceDetector(ConvergenceConfig(
            criteria=[ConvergenceCriterion.INERTIA],
            thresholds={},
        ))

    # Test invalid numeric parameters
    with pytest.raises(ValueError):
        detector = ConvergenceDetector(ConvergenceConfig(
            criteria=[ConvergenceCriterion.INERTIA],
            thresholds={ConvergenceCriterion.INERTIA: 1e-4},
            patience=0,  # Must be positive
        ))

    # Test invalid parameter relationships
    with pytest.raises(ValueError, match="min_iterations must be less than"):
        detector = ConvergenceDetector(ConvergenceConfig(
            criteria=[ConvergenceCriterion.INERTIA],
            thresholds={ConvergenceCriterion.INERTIA: 1e-4},
            min_iterations=100,
            max_iterations=50,
        ))


def test_convergence_state_initialization():
    """Test initialization of convergence state."""
    config = ConvergenceConfig(
        criteria=[
            ConvergenceCriterion.INERTIA,
            ConvergenceCriterion.STABILITY,
        ],
        thresholds={
            ConvergenceCriterion.INERTIA: 1e-4,
            ConvergenceCriterion.STABILITY: 0.95,
        },
    )
    detector = ConvergenceDetector(config)
    state = detector.initialize()

    assert state.iteration == 0
    assert not state.converged
    assert state.convergence_reason is None
    assert state.iterations_without_improvement == 0

    # Check metric initialization
    for criterion in config.criteria:
        assert criterion in state.metrics
        assert criterion in state.best_metrics
        assert criterion in state.history
        assert state.metrics[criterion] == float('inf')
        assert state.best_metrics[criterion] == float('inf')
        assert state.history[criterion] == []


def test_convergence_detection():
    """Test convergence detection with KMeans."""
    # Create sample data
    rng = np.random.RandomState(42)
    X = np.concatenate([
        rng.normal(0, 1, (100, 2)),
        rng.normal(4, 1, (100, 2)),
    ])

    # Configure convergence detection
    config = ConvergenceConfig(
        criteria=[ConvergenceCriterion.INERTIA],
        thresholds={ConvergenceCriterion.INERTIA: 1e-4},
        patience=3,
        min_iterations=2,
        max_iterations=10,
        window_size=2,
    )
    detector = ConvergenceDetector(config)
    state = detector.initialize()

    # Run clustering with convergence detection
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
    while not state.converged and state.iteration < config.max_iterations:
        kmeans.fit(X)
        state = detector.update(kmeans, X, state)

    # Check convergence
    assert state.converged
    assert state.convergence_reason is not None
    assert 0 < state.iteration <= config.max_iterations


def test_multiple_criteria():
    """Test convergence detection with multiple criteria."""
    # Create sample data
    rng = np.random.RandomState(42)
    X = np.concatenate([
        rng.normal(0, 1, (100, 2)),
        rng.normal(4, 1, (100, 2)),
    ])

    # Configure convergence detection with multiple criteria
    config = ConvergenceConfig(
        criteria=[
            ConvergenceCriterion.INERTIA,
            ConvergenceCriterion.SILHOUETTE,
        ],
        thresholds={
            ConvergenceCriterion.INERTIA: 1e-4,
            ConvergenceCriterion.SILHOUETTE: -0.5,  # Negative because we minimize
        },
        window_size=2,
    )
    detector = ConvergenceDetector(config)
    state = detector.initialize()

    # Run clustering
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
    while not state.converged and state.iteration < config.max_iterations:
        kmeans.fit(X)
        state = detector.update(kmeans, X, state)

    # Check that metrics were tracked
    for criterion in config.criteria:
        assert len(state.history[criterion]) == state.iteration
        assert all(isinstance(x, float) for x in state.history[criterion])


def test_early_stopping():
    """Test early stopping with patience."""
    # Create sample data with no clear clusters
    rng = np.random.RandomState(42)
    X = rng.uniform(-1, 1, (100, 2))

    # Configure convergence detection with short patience
    config = ConvergenceConfig(
        criteria=[ConvergenceCriterion.INERTIA],
        thresholds={ConvergenceCriterion.INERTIA: 1e-10},  # Very strict
        patience=2,  # Short patience
        min_iterations=2,
        window_size=2,
    )
    detector = ConvergenceDetector(config)
    state = detector.initialize()

    # Run clustering
    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=1)
    while not state.converged and state.iteration < config.max_iterations:
        kmeans.fit(X)
        state = detector.update(kmeans, X, state)

    # Check early stopping
    assert state.converged
    assert "No improvement" in state.convergence_reason
    assert state.iterations_without_improvement >= config.patience


def test_custom_metric():
    """Test convergence detection with custom metric."""
    def custom_metric(estimator, X):
        """Simple custom metric based on cluster sizes."""
        labels = estimator.labels_
        sizes = np.bincount(labels)
        return np.std(sizes) / np.mean(sizes)  # Size variation coefficient

    # Create sample data
    rng = np.random.RandomState(42)
    X = np.concatenate([
        rng.normal(0, 1, (100, 2)),
        rng.normal(4, 1, (100, 2)),
    ])

    # Configure convergence detection with custom metric
    config = ConvergenceConfig(
        criteria=[ConvergenceCriterion.CUSTOM],
        thresholds={ConvergenceCriterion.CUSTOM: 0.1},
        custom_metric=custom_metric,
        window_size=2,
    )
    detector = ConvergenceDetector(config)
    state = detector.initialize()

    # Run clustering
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
    while not state.converged and state.iteration < config.max_iterations:
        kmeans.fit(X)
        state = detector.update(kmeans, X, state)

    # Check custom metric tracking
    assert len(state.history[ConvergenceCriterion.CUSTOM]) == state.iteration
    assert all(isinstance(x, float) for x in state.history[ConvergenceCriterion.CUSTOM])