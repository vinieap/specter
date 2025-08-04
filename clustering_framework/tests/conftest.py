"""
Test fixtures for clustering framework.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def blobs_data():
    """Generate well-separated Gaussian blobs."""
    X, y = make_blobs(
        n_samples=300,
        n_features=2,
        centers=4,
        cluster_std=1.0,
        random_state=42
    )
    return StandardScaler().fit_transform(X), y


@pytest.fixture
def moons_data():
    """Generate two interleaving half circles."""
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    return StandardScaler().fit_transform(X), y


@pytest.fixture
def circles_data():
    """Generate concentric circles."""
    X, y = make_circles(n_samples=200, noise=0.1, random_state=42)
    return StandardScaler().fit_transform(X), y