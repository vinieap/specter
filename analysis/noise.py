"""Noise analysis module for clustering results.

This module provides tools for analyzing and classifying noise points in clustering results,
along with recommendations for handling noisy data.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

@dataclass
class NoiseAnalysis:
    """Structured noise analysis results.
    
    Attributes:
        noise_ratio: Percentage of points classified as noise
        noise_indices: Array of indices for noise points
        noise_types: Dictionary mapping noise types to arrays of point indices
        recommendations: Dictionary of recommendations for handling each noise type
    """
    noise_ratio: float
    noise_indices: np.ndarray
    noise_types: Dict[str, np.ndarray]
    recommendations: Dict[str, List[str]]

class NoiseAnalyzer:
    """Analyzes noise points in clustering results."""
    
    def __init__(self, n_neighbors: int = 5, contamination: float = 0.1):
        """Initialize the noise analyzer.
        
        Args:
            n_neighbors: Number of neighbors to consider for density estimation
            contamination: Expected proportion of noise points
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self._nn = None
        self._densities = None
    
    def fit(self, X: np.ndarray) -> 'NoiseAnalyzer':
        """Fit the analyzer to the data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            self: The fitted analyzer
        """
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self._nn.fit(X)
        distances, _ = self._nn.kneighbors()
        self._densities = np.mean(distances, axis=1)
        return self
    
    def analyze(self, X: np.ndarray, labels: np.ndarray) -> NoiseAnalysis:
        """Analyze noise points in clustering results.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            labels: Cluster labels including noise (-1)
            
        Returns:
            NoiseAnalysis object containing analysis results
        """
        if self._nn is None:
            self.fit(X)
            
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]
        noise_ratio = len(noise_indices) / len(X)
        
        # Classify noise points
        noise_types = self._classify_noise_points(X, noise_indices)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(noise_types)
        
        return NoiseAnalysis(
            noise_ratio=noise_ratio,
            noise_indices=noise_indices,
            noise_types=noise_types,
            recommendations=recommendations
        )
    
    def _classify_noise_points(self, X: np.ndarray, noise_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Classify noise points into different types.
        
        Types:
        - outliers: Points far from any cluster
        - boundary: Points between clusters
        - sparse: Points in low-density regions
        """
        if len(noise_indices) == 0:
            return {"outliers": np.array([], dtype=int),
                    "boundary": np.array([], dtype=int),
                    "sparse": np.array([], dtype=int)}
        
        noise_densities = self._densities[noise_indices]
        density_threshold = np.percentile(self._densities, (1 - self.contamination) * 100)
        
        # Classify based on density
        outliers = noise_indices[noise_densities > density_threshold * 2]
        boundary = noise_indices[
            (noise_densities <= density_threshold * 2) & 
            (noise_densities > density_threshold)
        ]
        sparse = noise_indices[noise_densities <= density_threshold]
        
        return {
            "outliers": outliers,
            "boundary": boundary,
            "sparse": sparse
        }
    
    def _generate_recommendations(self, noise_types: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Generate recommendations for handling each type of noise."""
        recommendations = {}
        
        if len(noise_types["outliers"]) > 0:
            recommendations["outliers"] = [
                "Consider removing these points as they may be anomalies",
                "Investigate these points for potential data quality issues",
                "If meaningful, consider creating a separate outlier cluster"
            ]
            
        if len(noise_types["boundary"]) > 0:
            recommendations["boundary"] = [
                "Consider adjusting cluster boundaries or density thresholds",
                "These points might indicate overlapping clusters",
                "Try different clustering algorithms that handle overlapping clusters"
            ]
            
        if len(noise_types["sparse"]) > 0:
            recommendations["sparse"] = [
                "Consider using density-based clustering algorithms",
                "Adjust density parameters in the current algorithm",
                "These points might represent a valid sparse cluster"
            ]
            
        return recommendations