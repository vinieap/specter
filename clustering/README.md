# Intelligent Clustering Library

A powerful and easy-to-use library for automated clustering optimization. This library helps you find the best clustering algorithm and parameters for your data, even if you're not a machine learning expert.

## 🌟 Key Features

- **Multiple Clustering Algorithms**: Support for all major clustering methods
- **Automatic Optimization**: Finds the best parameters automatically
- **Smart Progress Tracking**: Real-time feedback on optimization
- **Noise Detection**: Identifies and handles noisy data
- **Convergence Detection**: Knows when optimal results are found
- **Helpful Recommendations**: Suggests best algorithms and parameters

## 📚 Quick Start

```python
from clustering.api import optimize_clustering
from sklearn.datasets import make_blobs

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Find the best clustering
results = optimize_clustering(
    X,
    algorithm="kmeans",  # Try different algorithms
    n_calls=50,         # Number of optimization attempts
    use_dashboard=True  # View progress in real-time
)

# Get the best clustering model
best_clusterer = results["best_clusterer"]
labels = best_clusterer.fit_predict(X)
```

## 🎯 Available Clustering Algorithms

### Basic Clustering
- **K-Means** (`"kmeans"`)
  - Fast and simple
  - Works best for round, equal-sized clusters
  - Good for: Well-separated, circular clusters

- **Mini-Batch K-Means** (`"mini_batch_kmeans"`)
  - Memory-efficient version of K-Means
  - Good for: Large datasets

- **Bisecting K-Means** (`"bisecting_kmeans"`)
  - Hierarchical version of K-Means
  - Good for: Hierarchical data structure

### Density-Based
- **DBSCAN** (`"dbscan"`)
  - Finds clusters of any shape
  - Handles noise automatically
  - Good for: Irregular shapes, noise

- **HDBSCAN** (`"hdbscan"`)
  - More robust version of DBSCAN
  - Automatically adapts to different densities
  - Good for: Variable density clusters

- **OPTICS** (`"optics"`)
  - Similar to DBSCAN but more flexible
  - Good for: Multiple density levels

### Hierarchical
- **Agglomerative** (`"agglomerative"`)
  - Builds clusters hierarchically
  - Multiple linking methods
  - Good for: Hierarchical relationships

- **BIRCH** (`"birch"`)
  - Memory-efficient hierarchical clustering
  - Good for: Large datasets

### Affinity/Similarity-Based
- **Affinity Propagation** (`"affinity_propagation"`)
  - Automatically finds number of clusters
  - Good for: Unknown number of clusters

- **Spectral** (`"spectral"`)
  - Works well with complex shapes
  - Good for: Non-circular clusters

- **Mean Shift** (`"mean_shift"`)
  - Finds clusters automatically
  - Good for: Natural cluster shapes

## 🚀 Advanced Features

### Noise Detection

The library automatically detects different types of noise in your data:

```python
from clustering.noise import NoiseDetector

detector = NoiseDetector()
noise_profile = detector.detect_noise(X)

print(f"Noise ratio: {noise_profile.noise_ratio:.2f}")
print("\nRecommendations:")
for category, recs in noise_profile.recommendations.items():
    print(f"\n{category}:")
    for rec in recs:
        print(f"  - {rec}")
```

Types of noise detected:
- Global Outliers (points far from everything)
- Local Outliers (points that don't fit locally)
- Density Noise (points in sparse areas)
- Bridge Points (points between clusters)

### Convergence Detection

The library monitors optimization progress and detects when the best results are found:

```python
results = optimize_clustering(
    X,
    algorithm="kmeans",
    n_calls=100,
    verbosity=2  # Show detailed progress
)

# Check convergence details
convergence = results["convergence_status"]
print(f"Converged: {convergence['converged']}")
print(f"Confidence: {convergence['confidence']:.2f}")
print(f"Method: {convergence['method']}")
```

## 📊 Progress Tracking

Watch the optimization progress in real-time:

```python
results = optimize_clustering(
    X,
    algorithm="spectral",
    n_calls=50,
    use_dashboard=True,  # Open web dashboard
    verbosity=2          # Show detailed progress
)
```

You'll see:
- Current best score
- Success rate
- Estimated time remaining
- Convergence status
- Parameter importance

## 🔧 Choosing the Right Algorithm

Here's a simple guide for choosing algorithms:

1. **Well-Separated, Round Clusters**
   - Try: K-Means, Mini-Batch K-Means
   ```python
   results = optimize_clustering(X, algorithm="kmeans")
   ```

2. **Irregular Shapes or Noise**
   - Try: DBSCAN, HDBSCAN
   ```python
   results = optimize_clustering(X, algorithm="hdbscan")
   ```

3. **Unknown Number of Clusters**
   - Try: Affinity Propagation, Mean Shift
   ```python
   results = optimize_clustering(X, algorithm="affinity_propagation")
   ```

4. **Large Datasets**
   - Try: Mini-Batch K-Means, BIRCH
   ```python
   results = optimize_clustering(X, algorithm="mini_batch_kmeans")
   ```

5. **Complex Shapes**
   - Try: Spectral Clustering, DBSCAN
   ```python
   results = optimize_clustering(X, algorithm="spectral")
   ```

## 🎛️ Optimization Modes

### Batch Mode (Default)
Best for:
- Large datasets
- Multiple CPU cores
- When memory is available

```python
results = optimize_clustering(
    X,
    use_batch_optimizer=True,  # Default
    n_jobs=4                   # Number of CPU cores
)
```

### Sequential Mode
Best for:
- Small datasets
- Memory constraints
- Debugging

```python
results = optimize_clustering(
    X,
    use_batch_optimizer=False,
    verbosity=2  # See detailed progress
)
```

## 📈 Visualization Example

```python
import matplotlib.pyplot as plt

# Get clustering results
results = optimize_clustering(X, algorithm="kmeans")
labels = results["best_clusterer"].fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f'Best Score: {results["best_score"]:.4f}')
plt.show()
```

## 🔍 Tips for Best Results

1. **Start Simple**
   - Try K-Means first
   - If results aren't good, try more complex algorithms

2. **Watch for Noise**
   - If noise_ratio > 0.1, use DBSCAN or HDBSCAN
   - Consider preprocessing to remove noise

3. **Monitor Convergence**
   - Higher verbosity shows convergence details
   - Stop early if convergence is confident

4. **Use the Dashboard**
   - Set use_dashboard=True for visual monitoring
   - Watch parameter importance evolve

5. **Adjust Parameters**
   - Start with default n_calls=100
   - Increase for complex datasets
   - Use batch mode for large datasets

## 🤝 Contributing

Contributions are welcome! Please check our contribution guidelines for details.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.