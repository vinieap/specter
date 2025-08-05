"""Clustering algorithms package."""

from .base import ClusteringAlgorithm, ParamDict
from .kmeans import KMeansAlgorithm
from .mini_batch_kmeans import MiniBatchKMeansAlgorithm
from .bisecting_kmeans import BisectingKMeansAlgorithm
from .dbscan import DBSCANAlgorithm
from .hdbscan import HDBSCANAlgorithm
from .optics import OPTICSAlgorithm
from .agglomerative import AgglomerativeClusteringAlgorithm
from .birch import BirchAlgorithm
from .affinity_propagation import AffinityPropagationAlgorithm
from .spectral import SpectralClusteringAlgorithm
from .mean_shift import MeanShiftAlgorithm

__all__ = [
    # Base classes
    "ClusteringAlgorithm",
    "ParamDict",
    # Basic clustering
    "KMeansAlgorithm",
    "MiniBatchKMeansAlgorithm",
    "BisectingKMeansAlgorithm",
    # Density-based
    "DBSCANAlgorithm",
    "HDBSCANAlgorithm",
    "OPTICSAlgorithm",
    # Hierarchical
    "AgglomerativeClusteringAlgorithm",
    "BirchAlgorithm",
    # Affinity/similarity-based
    "AffinityPropagationAlgorithm",
    "SpectralClusteringAlgorithm",
    "MeanShiftAlgorithm",
]
