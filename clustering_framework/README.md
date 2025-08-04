# Clustering Framework

A comprehensive framework for clustering optimization and analysis, supporting multiple algorithms and providing robust evaluation tools.

## Features

- **Multiple Clustering Algorithms**
  - K-means for well-separated clusters
  - DBSCAN for density-based clustering
  - Spectral Clustering for complex shapes
  
- **Automated Parameter Optimization**
  - Algorithm-specific parameter tuning
  - Score tracking and model selection
  - Customizable number of optimization iterations

- **Cluster Analysis Tools**
  - Stability analysis through cross-validation
  - Noise sensitivity analysis
  - Multiple evaluation metrics
    - Silhouette score
    - Calinski-Harabasz index
    - Davies-Bouldin index

- **Flexible API**
  - Quick clustering for simple tasks
  - Detailed optimization for fine-tuning
  - Comprehensive analysis capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- conda (recommended for environment management)

### Setting up the environment

```bash
# Create and activate conda environment
conda create -n spec_opt python=3.12
conda activate spec_opt

# Install the package in development mode
cd clustering_framework
pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from clustering_framework import quick_cluster

# Generate sample data
X = np.random.randn(100, 2)

# Quick clustering with automatic evaluation
model, metrics = quick_cluster(X, n_clusters=3)
print(f"Silhouette score: {metrics['silhouette']:.3f}")
```

### Advanced Usage

```python
from clustering_framework import optimize_clustering, analyze_clusters, evaluate_clustering

# Optimize clustering parameters
results = optimize_clustering(
    X,
    algorithm="kmeans",
    n_calls=50,
    random_state=42,
    n_clusters=3
)

# Analyze cluster stability and noise sensitivity
analysis = analyze_clusters(
    X,
    results.best_model,
    noise_analysis=True,
    stability_analysis=True
)

# Evaluate with multiple metrics
metrics = evaluate_clustering(
    X,
    results.best_model,
    metrics=["silhouette", "calinski_harabasz", "davies_bouldin"]
)
```

### Algorithm-Specific Examples

#### K-means Clustering
```python
# For well-separated, spherical clusters
model, metrics = quick_cluster(
    X,
    algorithm="kmeans",
    n_clusters=4,
    init="k-means++",
    max_iter=300
)
```

#### DBSCAN Clustering
```python
# For density-based clustering
model, metrics = quick_cluster(
    X,
    algorithm="dbscan",
    eps=0.3,
    min_samples=5
)
```

#### Spectral Clustering
```python
# For complex shapes like concentric circles
model, metrics = quick_cluster(
    X,
    algorithm="spectral",
    n_clusters=2,
    affinity="nearest_neighbors"
)
```

## Project Structure

```
clustering_framework/
├── src/
│   └── clustering_framework/
│       ├── algorithms/     # Clustering algorithm implementations
│       ├── analysis/      # Analysis tools
│       │   ├── stability.py   # Stability analysis
│       │   └── noise.py       # Noise sensitivity analysis
│       ├── core/         # Core functionality
│       │   ├── api.py        # Main API functions
│       │   ├── algorithm.py  # Algorithm management
│       │   └── optimizer.py  # Parameter optimization
│       └── utils/        # Utility functions
│           └── metrics.py    # Evaluation metrics
├── tests/               # Comprehensive test suite
├── docs/               # Documentation
└── examples/           # Usage examples
```

## Testing

The framework includes a comprehensive test suite covering:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file with verbose output
python -m pytest tests/test_api.py -v

# Run tests with coverage report
python -m pytest --cov=clustering_framework tests/
```

Test coverage includes:
- Different clustering algorithms
- Parameter optimization
- Analysis tools
- Various data types (blobs, moons, circles)
- Edge cases and error handling
