"""
Clustering optimizer implementation.
"""

from typing import Type, Dict, Any, List, Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, SpectralClustering


class ClusteringOptimizer:
    """Optimizer for clustering parameters."""

    def __init__(
        self,
        algorithm: Type[BaseEstimator],
        n_calls: int = 50,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize optimizer.

        Args:
            algorithm: Clustering algorithm class
            n_calls: Number of optimization iterations
            random_state: Random seed
            **kwargs: Additional parameters for the algorithm
        """
        self.algorithm = algorithm
        self.n_calls = n_calls
        self.random_state = random_state
        self.base_params = kwargs

        self.best_score: float = -np.inf
        self.best_model: Optional[BaseEstimator] = None
        self.history: List[Dict[str, Any]] = []

    def _evaluate_model(self, model: BaseEstimator, X: np.ndarray) -> float:
        """
        Evaluate clustering model.

        Args:
            model: Fitted clustering model
            X: Input data matrix

        Returns:
            Score value
        """
        from ..utils.metrics import compute_metrics

        # Get labels based on model type
        if isinstance(model, (SpectralClustering, DBSCAN)):
            labels = model.labels_
        else:
            labels = model.predict(X)

        metrics = compute_metrics(X, labels, ["silhouette"])
        return metrics["silhouette"]

    def fit(self, X: np.ndarray) -> "ClusteringOptimizer":
        """
        Run optimization.

        Args:
            X: Input data matrix

        Returns:
            Self
        """
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_calls):
            # For simplicity, just try different random states or parameters
            current_params = self.base_params.copy()

            # Handle algorithm-specific optimization
            if self.algorithm == DBSCAN:
                # For DBSCAN, vary eps and min_samples
                current_params["eps"] = rng.uniform(
                    0.1, 0.5
                )  # Reduced range for better clustering
                current_params["min_samples"] = rng.randint(3, 8)  # Adjusted range
            elif self.algorithm == SpectralClustering:
                # For SpectralClustering, vary n_neighbors and eigen_solver
                current_params["random_state"] = rng.randint(0, 10000)
                if (
                    "affinity" in current_params
                    and current_params["affinity"] == "nearest_neighbors"
                ):
                    current_params["n_neighbors"] = rng.randint(5, 15)
            else:
                # For other algorithms (like KMeans), just vary random_state
                current_params["random_state"] = rng.randint(0, 10000)

            # Fit and evaluate model
            model = self.algorithm(**current_params)
            model.fit(X)
            score = self._evaluate_model(model, X)

            # Track results
            self.history.append(
                {"iteration": i, "params": current_params, "score": score}
            )

            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_model = model

        return self
