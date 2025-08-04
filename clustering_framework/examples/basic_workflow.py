"""
Example demonstrating a basic clustering workflow.
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from clustering_framework import (
    quick_cluster,
    analyze_and_improve,
    find_optimal_clusters
)

# Generate sample data
X, true_labels = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Find optimal number of clusters
n_clusters = find_optimal_clusters(
    X,
    max_clusters=10,
    algorithm='kmeans'
)
print(f"Optimal number of clusters: {n_clusters}")

# Perform clustering
model, metrics = quick_cluster(
    X,
    n_clusters=n_clusters,
    algorithm='kmeans'
)
print(f"Initial silhouette score: {metrics['silhouette']:.3f}")

# Analyze and improve results
improved_model, analysis = analyze_and_improve(
    X,
    model
)

# Print recommendations
print("\nAnalysis recommendations:")
for rec in analysis['recommendations']:
    print(f"- {rec}")

# Visualize results
plt.figure(figsize=(12, 5))

# Original clustering
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis')
plt.title('Original Clustering')

# Improved clustering
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=improved_model.labels_, cmap='viridis')
plt.title('Improved Clustering')

plt.tight_layout()
plt.show()