"""Noise detection and analysis for clustering data."""
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


@dataclass
class NoiseProfile:
    """Profile of detected noise in the data."""
    noise_ratio: float  # Ratio of noise points to total points
    noise_indices: np.ndarray  # Indices of noise points
    noise_types: Dict[str, np.ndarray]  # Different types of noise points
    noise_scores: np.ndarray  # Noise scores for each point
    recommendations: Dict[str, Any]  # Recommendations for handling noise
    details: Dict[str, Any]  # Detailed analysis results


class NoiseDetector:
    """
    Advanced noise detection for clustering data.
    
    Features:
    - Multiple noise detection methods
    - Noise type classification
    - Local and global noise analysis
    - Density-based noise detection
    - Statistical outlier detection
    - Ensemble noise detection
    - Recommendations for noise handling
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        random_state: int = 42,
    ):
        """Initialize noise detector.

        Parameters
        ----------
        contamination : float, default=0.1
            Expected proportion of noise points
        n_neighbors : int, default=20
            Number of neighbors for local noise detection
        random_state : int, default=42
            Random state for reproducible results
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def detect_noise(self, X: np.ndarray) -> NoiseProfile:
        """Detect and analyze noise in the data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)

        Returns
        -------
        NoiseProfile
            Detailed noise analysis results
        """
        # Scale the data for consistent noise detection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run all noise detection methods
        noise_detectors = [
            self._detect_statistical_outliers,
            self._detect_density_based_noise,
            self._detect_isolation_forest_noise,
            self._detect_local_outliers,
            self._detect_covariance_based_noise,
        ]

        # Collect results from all methods
        noise_votes = np.zeros(len(X))
        all_noise_scores = []
        method_results = {}

        for detector in noise_detectors:
            noise_mask, noise_scores, details = detector(X_scaled)
            noise_votes[noise_mask] += 1
            all_noise_scores.append(noise_scores)
            method_results[detector.__name__] = details

        # Combine noise scores using weighted average
        combined_scores = np.mean(all_noise_scores, axis=0)

        # Classify noise types
        noise_types = self._classify_noise_types(X_scaled, combined_scores, method_results)

        # Generate final noise mask using ensemble voting
        ensemble_threshold = len(noise_detectors) / 2  # More than 50% agreement
        final_noise_mask = noise_votes >= ensemble_threshold

        # Calculate noise ratio
        noise_ratio = np.mean(final_noise_mask)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            X, noise_ratio, noise_types, method_results
        )

        return NoiseProfile(
            noise_ratio=noise_ratio,
            noise_indices=np.where(final_noise_mask)[0],
            noise_types=noise_types,
            noise_scores=combined_scores,
            recommendations=recommendations,
            details=method_results,
        )

    def _detect_statistical_outliers(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Detect outliers using statistical methods."""
        # Calculate Z-scores for each feature
        z_scores = stats.zscore(X, axis=0)
        
        # Calculate Mahalanobis distances
        cov = np.cov(X.T)
        inv_cov = np.linalg.pinv(cov)
        mean = np.mean(X, axis=0)
        mahalanobis_dist = np.sqrt(
            np.sum(
                np.dot(X - mean, inv_cov) * (X - mean),
                axis=1
            )
        )
        
        # Combine evidence
        max_z_scores = np.max(np.abs(z_scores), axis=1)
        noise_scores = (
            0.7 * stats.norm.cdf(max_z_scores) +
            0.3 * stats.chi2.cdf(mahalanobis_dist, df=X.shape[1])
        )
        
        # Determine noise points
        threshold = np.percentile(noise_scores, (1 - self.contamination) * 100)
        noise_mask = noise_scores > threshold

        return noise_mask, noise_scores, {
            "z_scores": z_scores,
            "mahalanobis_dist": mahalanobis_dist,
            "threshold": threshold,
        }

    def _detect_density_based_noise(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Detect noise using density-based methods."""
        # Calculate local density using nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Calculate density scores
        density_scores = np.mean(distances, axis=1)
        reachability = np.max(distances, axis=1)
        
        # Calculate local density ratio
        k_distances = distances[:, -1]
        density_ratio = k_distances / np.mean(distances, axis=1)
        
        # Combine metrics
        noise_scores = (
            0.4 * stats.norm.cdf(density_scores) +
            0.3 * stats.norm.cdf(reachability) +
            0.3 * stats.norm.cdf(density_ratio)
        )
        
        # Determine noise points
        threshold = np.percentile(noise_scores, (1 - self.contamination) * 100)
        noise_mask = noise_scores > threshold

        return noise_mask, noise_scores, {
            "density_scores": density_scores,
            "reachability": reachability,
            "density_ratio": density_ratio,
            "threshold": threshold,
        }

    def _detect_isolation_forest_noise(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Detect noise using Isolation Forest."""
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        
        # Fit and predict
        iso_forest.fit(X)
        scores = iso_forest.score_samples(X)
        
        # Convert scores to probabilities (higher means more likely to be noise)
        noise_scores = 1 - stats.norm.cdf(scores)
        noise_mask = iso_forest.predict(X) == -1

        return noise_mask, noise_scores, {
            "isolation_scores": scores,
            "n_estimators": iso_forest.n_estimators_,
        }

    def _detect_local_outliers(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Detect noise using Local Outlier Factor."""
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )
        
        # Fit and predict
        lof.fit(X)
        scores = -lof.score_samples(X)  # Negative scores for consistency
        
        # Convert scores to probabilities
        noise_scores = stats.norm.cdf(scores)
        noise_mask = lof.predict(X) == -1

        return noise_mask, noise_scores, {
            "lof_scores": scores,
            "negative_outlier_factor": lof.negative_outlier_factor_,
        }

    def _detect_covariance_based_noise(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Detect noise using robust covariance estimation."""
        ee = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        
        # Fit and predict
        ee.fit(X)
        scores = ee.score_samples(X)
        
        # Convert scores to probabilities
        noise_scores = 1 - stats.norm.cdf(scores)
        noise_mask = ee.predict(X) == -1

        return noise_mask, noise_scores, {
            "mahalanobis_scores": scores,
            "location": ee.location_,
            "precision": ee.precision_,
        }

    def _classify_noise_types(
        self,
        X: np.ndarray,
        noise_scores: np.ndarray,
        method_results: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Classify different types of noise points."""
        noise_types = {}
        
        # Global outliers (far from all clusters)
        global_outliers = noise_scores > np.percentile(noise_scores, 95)
        noise_types["global_outliers"] = global_outliers
        
        # Local outliers (outliers relative to nearest cluster)
        local_outlier_scores = method_results["_detect_local_outliers"]["lof_scores"]
        local_outliers = (
            (local_outlier_scores > np.percentile(local_outlier_scores, 90)) &
            ~global_outliers  # Not already classified as global outliers
        )
        noise_types["local_outliers"] = local_outliers
        
        # Density-based noise (points in sparse regions)
        density_scores = method_results["_detect_density_based_noise"]["density_scores"]
        density_noise = (
            (density_scores > np.percentile(density_scores, 90)) &
            ~global_outliers &
            ~local_outliers
        )
        noise_types["density_noise"] = density_noise
        
        # Bridge points (points between clusters)
        reachability = method_results["_detect_density_based_noise"]["reachability"]
        density_ratio = method_results["_detect_density_based_noise"]["density_ratio"]
        bridge_points = (
            (reachability > np.percentile(reachability, 80)) &
            (density_ratio < np.percentile(density_ratio, 50)) &
            ~global_outliers &
            ~local_outliers &
            ~density_noise
        )
        noise_types["bridge_points"] = bridge_points

        return noise_types

    def _generate_recommendations(
        self,
        X: np.ndarray,
        noise_ratio: float,
        noise_types: Dict[str, np.ndarray],
        method_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate recommendations for handling noise."""
        recommendations = {}
        
        # Algorithm recommendations
        if noise_ratio > 0.2:
            recommendations["algorithm"] = [
                "DBSCAN",
                "HDBSCAN",
                "Robust methods like BIRCH"
            ]
        elif noise_ratio > 0.1:
            recommendations["algorithm"] = [
                "DBSCAN",
                "Spectral Clustering with RBF kernel"
            ]
        else:
            recommendations["algorithm"] = [
                "Any standard clustering algorithm",
                "Consider K-Means or Agglomerative"
            ]
        
        # Parameter recommendations
        param_recommendations = {}
        
        if "DBSCAN" in recommendations["algorithm"][0]:
            # Calculate eps based on noise distribution
            distances = method_results["_detect_density_based_noise"]["density_scores"]
            suggested_eps = np.percentile(distances, 85)
            param_recommendations["eps"] = suggested_eps
            param_recommendations["min_samples"] = max(
                2,
                int(np.log(len(X)) * (1 - noise_ratio))
            )
        
        if "Spectral" in str(recommendations["algorithm"]):
            param_recommendations["gamma"] = 1 / (X.shape[1] * X.var())
        
        recommendations["parameters"] = param_recommendations
        
        # Preprocessing recommendations
        preprocess_recommendations = []
        
        if noise_ratio > 0.15:
            preprocess_recommendations.append(
                "Consider removing global outliers before clustering"
            )
        
        if len(noise_types["bridge_points"]) > 0.05 * len(X):
            preprocess_recommendations.append(
                "Consider feature scaling or dimensionality reduction"
            )
        
        if len(noise_types["density_noise"]) > 0.1 * len(X):
            preprocess_recommendations.append(
                "Consider density-based preprocessing or weighted features"
            )
        
        recommendations["preprocessing"] = preprocess_recommendations
        
        return recommendations