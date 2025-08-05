"""Configuration management for the clustering framework.

This module provides a centralized configuration system for managing framework
parameters, constants, and default values. It uses a hierarchical structure
to organize settings by component and provides type validation.
"""

from dataclasses import dataclass
from typing import Dict, Any, ClassVar


@dataclass
class OptimizerConfig:
    """Configuration for optimization parameters.

    Attributes:
        default_n_startup_trials: Default number of random trials before optimization
        default_patience: Default number of trials without improvement before convergence
        default_min_delta: Minimum change in score to be considered improvement
        default_min_trials: Minimum number of trials before allowing convergence
        default_dashboard_port: Default port for Optuna dashboard
        score_worse_than_random: Score threshold below which model is considered worse than random
        max_consecutive_failures: Maximum number of consecutive failed trials before warning
    """

    default_n_startup_trials: int = 20
    default_patience: int = 10
    default_min_delta: float = 1e-4
    default_min_trials: int = 20
    default_dashboard_port: int = 8080
    score_worse_than_random: float = -0.1
    max_consecutive_failures: int = 5


@dataclass
class KMeansConfig:
    """Configuration for K-means clustering algorithm.

    Attributes:
        max_clusters: Maximum number of clusters
        min_clusters: Minimum number of clusters
        max_init: Maximum number of initializations
        min_init: Minimum number of initializations
    """

    max_clusters: int = 20
    min_clusters: int = 2
    max_init: int = 20
    min_init: int = 5


@dataclass
class DBSCANConfig:
    """Configuration for DBSCAN clustering algorithm.

    Attributes:
        max_eps: Maximum epsilon value
        min_eps: Minimum epsilon value
        max_min_samples: Maximum min_samples value
        min_min_samples: Minimum min_samples value
    """

    max_eps: float = 0.5
    min_eps: float = 0.2
    max_min_samples: int = 8
    min_min_samples: int = 3


@dataclass
class SpectralConfig:
    """Configuration for Spectral clustering algorithm.

    Attributes:
        max_n_neighbors: Maximum number of neighbors
        min_n_neighbors: Minimum number of neighbors
        max_gamma: Maximum gamma value for RBF kernel
        min_gamma: Minimum gamma value for RBF kernel
    """

    max_n_neighbors: int = 15
    min_n_neighbors: int = 5
    max_gamma: float = 10.0
    min_gamma: float = 0.1


@dataclass
class AlgorithmConfig:
    """Configuration for all clustering algorithms.

    Attributes:
        kmeans: K-means specific configuration
        dbscan: DBSCAN specific configuration
        spectral: Spectral clustering specific configuration
    """

    kmeans: KMeansConfig = None
    dbscan: DBSCANConfig = None
    spectral: SpectralConfig = None

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.kmeans is None:
            self.kmeans = KMeansConfig()
        if self.dbscan is None:
            self.dbscan = DBSCANConfig()
        if self.spectral is None:
            self.spectral = SpectralConfig()


@dataclass
class AnalysisConfig:
    """Configuration for analysis tools.

    Attributes:
        stability_threshold: Threshold for considering clusters stable
        max_noise_ratio: Maximum acceptable ratio of noise points
        min_cluster_size: Minimum number of points for a valid cluster
        n_stability_runs: Number of runs for stability analysis
        noise_std_levels: Standard deviation levels for noise analysis
    """

    stability_threshold: float = 0.95
    max_noise_ratio: float = 0.1
    min_cluster_size: int = 5
    n_stability_runs: int = 5
    noise_std_levels: tuple = (0.01, 0.05, 0.1)


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics.

    Attributes:
        available_metrics: Dictionary of available metric names and their functions
        default_metric: Default metric for optimization
        metric_requires_min_clusters: Metrics that require at least 2 clusters
        invalid_score: Score to return when clustering is invalid
    """

    available_metrics: ClassVar[Dict[str, str]] = {
        "silhouette": "silhouette_score",
        "calinski_harabasz": "calinski_harabasz_score",
        "davies_bouldin": "davies_bouldin_score",
    }
    default_metric: str = "silhouette"
    metric_requires_min_clusters: ClassVar[tuple] = (
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    )
    invalid_score: float = float("-inf")


@dataclass
class FrameworkConfig:
    """Global framework configuration.

    This class provides a centralized configuration for all framework components.
    It can be instantiated with custom values or used with defaults.

    Attributes:
        optimizer: Optimizer configuration
        algorithm: Algorithm configuration
        analysis: Analysis configuration
        metrics: Metrics configuration
    """

    optimizer: OptimizerConfig = None
    algorithm: AlgorithmConfig = None
    analysis: AnalysisConfig = None
    metrics: MetricsConfig = None

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.algorithm is None:
            self.algorithm = AlgorithmConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()
        if self.metrics is None:
            self.metrics = MetricsConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FrameworkConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            FrameworkConfig instance with specified values
        """
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {}))
        algorithm_config = AlgorithmConfig(**config_dict.get("algorithm", {}))
        analysis_config = AnalysisConfig(**config_dict.get("analysis", {}))
        metrics_config = MetricsConfig(**config_dict.get("metrics", {}))

        return cls(
            optimizer=optimizer_config,
            algorithm=algorithm_config,
            analysis=analysis_config,
            metrics=metrics_config,
        )


# Global configuration instance
config = FrameworkConfig()
