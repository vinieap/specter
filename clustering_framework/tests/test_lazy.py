"""Tests for lazy loading functionality."""

import pytest

from clustering_framework.core.lazy import LazyLoader
from clustering_framework.algorithms import KMeansAlgorithm


def test_lazy_loader_creation():
    """Test creating a LazyLoader instance."""
    loader = LazyLoader(KMeansAlgorithm)
    assert loader is not None
    assert loader._algorithm_class == KMeansAlgorithm


def test_parameter_validation():
    """Test parameter validation without instance creation."""
    loader = LazyLoader(KMeansAlgorithm)

    # Test valid parameters
    valid_params = {"n_clusters": 3, "init": "k-means++"}
    assert loader.validate_parameters(valid_params) is True

    # Test invalid parameters
    invalid_params = {"n_clusters": -1}  # Invalid number of clusters
    assert loader.validate_parameters(invalid_params) is False

    # Test validation caching
    assert loader.validate_parameters(valid_params) is True  # Should use cache
    assert loader.validate_parameters(invalid_params) is False  # Should use cache


def test_instance_creation():
    """Test creating algorithm instances."""
    loader = LazyLoader(KMeansAlgorithm)

    # Create instance with valid parameters
    with loader.get_instance(random_state=42, n_clusters=3) as instance:
        assert isinstance(instance, KMeansAlgorithm)
        assert instance._parameters["n_clusters"] == 3

    # Test error on invalid parameters
    with pytest.raises(ValueError):
        with loader.get_instance(random_state=42, n_clusters=-1):
            pass


def test_validation_hook():
    """Test custom validation hook."""

    def validation_hook(params):
        if "n_clusters" in params and params["n_clusters"] > 10:
            raise ValueError("Too many clusters")

    loader = LazyLoader(KMeansAlgorithm, validation_hook=validation_hook)

    # Test valid parameters
    assert loader.validate_parameters({"n_clusters": 5}) is True

    # Test invalid parameters
    assert loader.validate_parameters({"n_clusters": 15}) is False


def test_validation_cache_management():
    """Test validation cache management."""
    loader = LazyLoader(KMeansAlgorithm)

    # Fill cache
    params = {"n_clusters": 3}
    loader.validate_parameters(params)
    assert len(loader._validation_cache) == 1

    # Clear cache
    loader.clear_validation_cache()
    assert len(loader._validation_cache) == 0


def test_validated_params_storage():
    """Test storage and retrieval of validated parameters."""
    loader = LazyLoader(KMeansAlgorithm)

    # Create instance with parameters
    params = {"n_clusters": 3, "init": "k-means++"}
    with loader.get_instance(random_state=42, **params) as instance:
        # Check stored parameters
        stored_params = loader.get_validated_params(42)
        assert stored_params is not None
        assert stored_params["n_clusters"] == 3
        assert stored_params["init"] == "k-means++"

    # Check non-existent random state
    assert loader.get_validated_params(999) is None
