# Clustering Framework

A comprehensive and robust Python framework for data clustering, optimization, and analysis. This framework provides a unified interface to multiple clustering algorithms, automated parameter optimization, and extensive evaluation tools. It is designed for both researchers and practitioners working with various types of datasets, from simple numerical data to complex high-dimensional structures.

## Overview

The Clustering Framework is built to address common challenges in data clustering:
- Finding optimal clustering parameters for your specific dataset
- Evaluating cluster quality and stability
- Handling different types of data structures and distributions
- Scaling to large datasets efficiently
- Providing reproducible and interpretable results

### Key Benefits
- **Unified Interface**: Consistent API across different clustering algorithms
- **Automated Optimization**: Smart parameter tuning for optimal results
- **Robust Evaluation**: Comprehensive metrics and analysis tools
- **Flexibility**: Support for various data types and clustering approaches
- **Scalability**: Efficient implementation for large datasets
- **Reproducibility**: Deterministic results with seed control

### When to Use This Framework
- **Data Exploration**: Quickly understand natural groupings in your data
- **Pattern Discovery**: Identify hidden structures and relationships
- **Customer Segmentation**: Group similar customers for targeted marketing
- **Image Segmentation**: Cluster pixels or features for image analysis
- **Anomaly Detection**: Identify outliers and unusual patterns
- **Document Clustering**: Group similar documents or text data

## Features

### Clustering Algorithms

#### K-means Clustering
- **Best for**: Well-separated, spherical clusters
- **Key Features**:
  - Fast and memory-efficient
  - Scalable to large datasets
  - Customizable initialization methods (k-means++, random)
- **Parameters**:
  - `n_clusters`: Number of clusters (required)
  - `init`: Initialization method ('k-means++', 'random')
  - `max_iter`: Maximum iterations for convergence
  - `n_init`: Number of times to run with different seeds
- **Use Cases**:
  - Customer segmentation
  - Image color quantization
  - Feature learning

#### DBSCAN (Density-Based Spatial Clustering)
- **Best for**: Clusters of varying shapes and densities
- **Key Features**:
  - No pre-defined number of clusters needed
  - Handles noise points automatically
  - Can find non-spherical clusters
- **Parameters**:
  - `eps`: Maximum distance between points in a cluster
  - `min_samples`: Minimum points to form a dense region
- **Use Cases**:
  - Spatial data analysis
  - Noise detection
  - Network clustering

#### Spectral Clustering
- **Best for**: Complex, non-spherical shapes
- **Key Features**:
  - Handles complex cluster shapes
  - Based on graph theory principles
  - Multiple affinity methods available
- **Parameters**:
  - `n_clusters`: Number of clusters
  - `affinity`: Similarity metric ('rbf', 'nearest_neighbors', etc.)
  - `n_neighbors`: Number of neighbors (for 'nearest_neighbors')
- **Use Cases**:
  - Image segmentation
  - Social network analysis
  - Manifold learning

### Automated Parameter Optimization

#### Bayesian Optimization
- **Features**:
  - Smart parameter space exploration
  - Efficient optimization strategy
  - Parallel optimization support
- **Customization Options**:
  - Number of optimization calls
  - Objective function selection
  - Parameter search spaces
  - Cross-validation strategy

#### Grid and Random Search
- **Features**:
  - Exhaustive parameter exploration
  - Random sampling for large spaces
  - Parallel execution support
- **Parameters**:
  - Search space definition
  - Number of iterations
  - Cross-validation folds

### Cluster Analysis Tools

#### Stability Analysis
- **Methods**:
  - Bootstrap resampling
  - Cross-validation
  - Noise injection
- **Metrics**:
  - Cluster consistency score
  - Label stability index
  - Membership probability

#### Quality Evaluation
- **Internal Metrics**:
  - Silhouette score (cluster separation)
  - Calinski-Harabasz index (density)
  - Davies-Bouldin index (cluster similarity)
- **External Metrics** (when ground truth available):
  - Adjusted Rand index
  - Normalized mutual information
  - V-measure

#### Noise Sensitivity Analysis
- **Features**:
  - Gaussian noise injection
  - Outlier impact analysis
  - Boundary stability assessment
- **Parameters**:
  - Noise levels
  - Number of iterations
  - Analysis metrics

### Flexible API

#### Quick Clustering Interface
```python
from clustering_framework import quick_cluster

# Simple clustering with automatic parameter selection
model, metrics = quick_cluster(
    data,
    algorithm="kmeans",
    n_clusters=3
)
```

#### Advanced Optimization Interface
```python
from clustering_framework import optimize_clustering

# Detailed parameter optimization
results = optimize_clustering(
    data,
    algorithm="spectral",
    n_calls=50,
    parameter_space={
        "n_clusters": [2, 3, 4, 5],
        "affinity": ["rbf", "nearest_neighbors"],
        "n_neighbors": [5, 10, 15]
    }
)
```

#### Analysis Interface
```python
from clustering_framework import analyze_clusters

# Comprehensive cluster analysis
analysis = analyze_clusters(
    data,
    model,
    stability_analysis=True,
    noise_analysis=True,
    metrics=["silhouette", "calinski_harabasz"]
)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- conda (recommended for environment management)
- BLAS/LAPACK libraries for linear algebra operations
- C++ compiler for building extensions (optional, for performance optimization)

### Dependencies

The framework requires the following main packages:
- numpy>=1.20.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- joblib>=1.0.0

### Installation Methods

#### 1. Using pip (Recommended for Users)

```bash
# Create and activate virtual environment
python -m venv clustering_env
source clustering_env/bin/activate  # On Windows: clustering_env\Scripts\activate

# Install from PyPI
pip install clustering-framework

# Install with optional dependencies for all features
pip install clustering-framework[all]
```

#### 2. Using conda (Recommended for Scientific Computing)

```bash
# Create new environment with required packages
conda create -n clustering_env python=3.12
conda activate clustering_env

# Install main package and dependencies
conda install -c conda-forge clustering-framework
```

#### 3. Development Installation (For Contributors)

```bash
# Clone the repository
git clone https://github.com/username/clustering_framework.git
cd clustering_framework

# Create and activate conda environment
conda create -n clustering_dev python=3.12
conda activate clustering_dev

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"
```

### Verifying Installation

```python
# Run this in Python to verify installation
import clustering_framework as cf
print(cf.__version__)

# Should print the version number without errors
```

### Platform-Specific Notes

#### Windows
- Ensure Microsoft Visual C++ Build Tools are installed
- Use Anaconda distribution for easier dependency management

#### Linux
- Install BLAS/LAPACK development libraries:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libblas-dev liblapack-dev

  # CentOS/RHEL
  sudo yum install blas-devel lapack-devel
  ```

#### macOS
- Install Command Line Tools:
  ```bash
  xcode-select --install
  ```
- Use Homebrew for BLAS/LAPACK:
  ```bash
  brew install openblas lapack
  ```

## Working with Real Datasets

This section provides practical examples using real-world datasets to demonstrate the framework's capabilities in different scenarios.

### 1. Customer Segmentation (E-commerce Dataset)

```python
import pandas as pd
from clustering_framework import quick_cluster, preprocess_data
from clustering_framework.visualization import plot_clusters

# Load and preprocess e-commerce data
data = pd.read_csv('customer_data.csv')
features = ['recency', 'frequency', 'monetary_value', 'avg_basket_size']

# Preprocess the data (scaling, handling missing values)
X = preprocess_data(
    data[features],
    scale=True,
    handle_missing='mean'
)

# Perform clustering with automatic parameter selection
model, metrics = quick_cluster(
    X,
    algorithm='kmeans',
    n_clusters_range=(3, 8),  # Try 3-8 clusters
    optimization_metric='silhouette'
)

# Analyze and visualize results
plot_clusters(X, model.labels_, features=features)
print(f"Number of clusters: {len(set(model.labels_))}")
print(f"Silhouette score: {metrics['silhouette']:.3f}")

# Get cluster profiles
cluster_profiles = pd.DataFrame({
    'Cluster': range(len(set(model.labels_))),
    'Size': pd.Series(model.labels_).value_counts().sort_index(),
    'Avg_Recency': data.groupby(model.labels_)['recency'].mean(),
    'Avg_Frequency': data.groupby(model.labels_)['frequency'].mean(),
    'Avg_Monetary': data.groupby(model.labels_)['monetary_value'].mean()
})
print("\nCluster Profiles:")
print(cluster_profiles)
```

### 2. Image Segmentation (Medical Imaging)

```python
import numpy as np
from skimage import io
from clustering_framework import optimize_clustering
from clustering_framework.preprocessing import extract_image_features
from clustering_framework.visualization import plot_segmentation

# Load and preprocess medical image
image = io.imread('brain_scan.png')
features = extract_image_features(
    image,
    feature_type=['intensity', 'texture', 'position']
)

# Optimize spectral clustering parameters
results = optimize_clustering(
    features,
    algorithm='spectral',
    parameter_space={
        'n_clusters': [3, 4, 5],
        'affinity': ['rbf', 'nearest_neighbors'],
        'n_neighbors': [10, 20, 30]
    },
    n_calls=30
)

# Apply best model and visualize
segmented_image = plot_segmentation(
    image,
    results.best_model.labels_,
    overlay=True
)

print("Best parameters:", results.best_params)
print("Clustering metrics:", results.best_metrics)
```

### 3. Text Document Clustering (News Articles)

```python
from clustering_framework import quick_cluster
from clustering_framework.preprocessing import text_to_features
from clustering_framework.visualization import plot_cluster_wordclouds

# Load and preprocess text data
docs = pd.read_csv('news_articles.csv')
X = text_to_features(
    docs['text'],
    method='tfidf',
    max_features=1000,
    stop_words='english'
)

# Perform clustering
model, metrics = quick_cluster(
    X,
    algorithm='kmeans',
    n_clusters=5,
    preprocessing='normalize'
)

# Analyze clusters
plot_cluster_wordclouds(
    docs['text'],
    model.labels_,
    top_n_words=20
)

# Print top terms per cluster
print("\nTop terms per cluster:")
for i, terms in enumerate(get_cluster_terms(model, feature_names)):
    print(f"\nCluster {i}: {', '.join(terms[:10])}")
```

### 4. Anomaly Detection (Network Traffic)

```python
import pandas as pd
from clustering_framework import analyze_clusters
from clustering_framework.algorithms import DBSCAN

# Load network traffic data
data = pd.read_csv('network_traffic.csv')
features = ['bytes_sent', 'bytes_received', 'duration', 'port']

# Preprocess and normalize features
X = preprocess_data(
    data[features],
    scale=True,
    handle_missing='drop'
)

# Apply DBSCAN for anomaly detection
model = DBSCAN(
    eps=0.3,
    min_samples=5,
    metric='euclidean'
)
model.fit(X)

# Analyze results
analysis = analyze_clusters(
    X,
    model,
    noise_analysis=True
)

# Print anomaly statistics
n_anomalies = sum(model.labels_ == -1)
print(f"\nDetected anomalies: {n_anomalies}")
print(f"Anomaly percentage: {n_anomalies/len(X)*100:.2f}%")

# Visualize anomalies
plot_anomalies(
    X,
    model.labels_,
    features=features[:2]  # Plot first two features
)
```

### 5. Time Series Clustering (Stock Market Data)

```python
from clustering_framework import optimize_clustering
from clustering_framework.preprocessing import extract_timeseries_features
from clustering_framework.visualization import plot_cluster_trends

# Load stock market data
data = pd.read_csv('stock_prices.csv')
stocks = data.pivot(
    index='date',
    columns='symbol',
    values='close'
)

# Extract time series features
features = extract_timeseries_features(
    stocks,
    features=['trend', 'seasonality', 'volatility']
)

# Optimize clustering
results = optimize_clustering(
    features,
    algorithm='kmeans',
    n_clusters_range=(3, 8),
    n_calls=30
)

# Visualize cluster trends
plot_cluster_trends(
    stocks,
    results.best_model.labels_,
    n_periods=30  # Show last 30 days
)

# Print cluster characteristics
print("\nCluster Characteristics:")
for i in range(len(set(results.best_model.labels_))):
    stocks_in_cluster = stocks.columns[results.best_model.labels_ == i]
    print(f"\nCluster {i} stocks: {', '.join(stocks_in_cluster)}")
```

Each example includes:
- Data loading and preprocessing
- Algorithm selection and parameter optimization
- Result visualization and analysis
- Interpretation of results

The examples demonstrate how to:
- Handle different types of data (numerical, text, images)
- Choose appropriate algorithms and parameters
- Preprocess data effectively
- Visualize and interpret results
- Extract meaningful insights from clusters
```

## API Reference

### Core Functions

#### quick_cluster
```python
def quick_cluster(
    data: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    n_clusters_range: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Tuple[BaseEstimator, Dict[str, float]]:
    """
    Quick clustering with automatic parameter selection.

    Parameters:
        data: Input data matrix (n_samples, n_features)
        algorithm: Clustering algorithm ('kmeans', 'dbscan', 'spectral')
        n_clusters: Fixed number of clusters
        n_clusters_range: Range of clusters to try (min, max)
        **kwargs: Additional algorithm-specific parameters

    Returns:
        model: Fitted clustering model
        metrics: Dictionary of evaluation metrics
    """
```

#### optimize_clustering
```python
def optimize_clustering(
    data: np.ndarray,
    algorithm: str,
    parameter_space: Dict[str, Any],
    n_calls: int = 50,
    random_state: Optional[int] = None,
    optimization_metric: str = "silhouette",
    n_jobs: int = -1
) -> OptimizationResult:
    """
    Optimize clustering parameters using Bayesian optimization.

    Parameters:
        data: Input data matrix
        algorithm: Clustering algorithm name
        parameter_space: Dictionary of parameters to optimize
        n_calls: Number of optimization iterations
        random_state: Random seed for reproducibility
        optimization_metric: Metric to optimize
        n_jobs: Number of parallel jobs

    Returns:
        OptimizationResult object containing:
            - best_model: Best fitted model
            - best_params: Best parameters found
            - best_metrics: Metrics for best model
            - optimization_history: Full optimization history
    """
```

#### analyze_clusters
```python
def analyze_clusters(
    data: np.ndarray,
    model: BaseEstimator,
    stability_analysis: bool = True,
    noise_analysis: bool = False,
    n_iterations: int = 100,
    random_state: Optional[int] = None
) -> ClusterAnalysis:
    """
    Perform comprehensive cluster analysis.

    Parameters:
        data: Input data matrix
        model: Fitted clustering model
        stability_analysis: Whether to perform stability analysis
        noise_analysis: Whether to perform noise sensitivity analysis
        n_iterations: Number of bootstrap iterations
        random_state: Random seed for reproducibility

    Returns:
        ClusterAnalysis object containing:
            - stability_scores: Cluster stability metrics
            - noise_sensitivity: Noise impact analysis
            - cluster_profiles: Statistical profiles of clusters
    """
```

### Preprocessing Functions

#### preprocess_data
```python
def preprocess_data(
    data: Union[np.ndarray, pd.DataFrame],
    scale: bool = True,
    handle_missing: str = 'mean',
    categorical_encoding: str = 'onehot',
    feature_selection: Optional[str] = None
) -> np.ndarray:
    """
    Preprocess data for clustering.

    Parameters:
        data: Input data
        scale: Whether to scale features
        handle_missing: Strategy for missing values
        categorical_encoding: Method for encoding categorical variables
        feature_selection: Feature selection method

    Returns:
        Preprocessed data matrix
    """
```

#### extract_timeseries_features
```python
def extract_timeseries_features(
    data: Union[np.ndarray, pd.DataFrame],
    features: List[str],
    window_size: Optional[int] = None
) -> np.ndarray:
    """
    Extract features from time series data.

    Parameters:
        data: Time series data
        features: List of features to extract
        window_size: Size of rolling window

    Returns:
        Feature matrix
    """
```

### Visualization Functions

#### plot_clusters
```python
def plot_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    features: Optional[List[str]] = None,
    plot_type: str = 'scatter',
    **kwargs
) -> None:
    """
    Visualize clustering results.

    Parameters:
        data: Input data matrix
        labels: Cluster labels
        features: Feature names for axes
        plot_type: Type of plot ('scatter', 'pca', 'tsne')
        **kwargs: Additional plotting parameters
    """
```

#### plot_optimization_history
```python
def plot_optimization_history(
    results: OptimizationResult,
    plot_type: str = 'convergence'
) -> None:
    """
    Visualize optimization progress.

    Parameters:
        results: OptimizationResult object
        plot_type: Type of plot ('convergence', 'parameter_importance')
    """
```

### Algorithm Classes

#### KMeans
```python
class KMeans(BaseClusteringAlgorithm):
    """
    Enhanced K-means clustering implementation.

    Parameters:
        n_clusters: Number of clusters
        init: Initialization method
        max_iter: Maximum iterations
        n_init: Number of initializations
        random_state: Random seed
    """
```

#### DBSCAN
```python
class DBSCAN(BaseClusteringAlgorithm):
    """
    Enhanced DBSCAN clustering implementation.

    Parameters:
        eps: Maximum distance between points
        min_samples: Minimum points in neighborhood
        metric: Distance metric
        n_jobs: Number of parallel jobs
    """
```

#### SpectralClustering
```python
class SpectralClustering(BaseClusteringAlgorithm):
    """
    Enhanced spectral clustering implementation.

    Parameters:
        n_clusters: Number of clusters
        affinity: Affinity type
        n_neighbors: Number of neighbors
        random_state: Random seed
    """
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

## Troubleshooting and FAQ

### Common Issues

#### 1. Memory Issues with Large Datasets

**Problem**: Out of memory errors when clustering large datasets.

**Solutions**:
- Use mini-batch K-means for large-scale clustering
- Reduce dimensionality using PCA or feature selection
- Increase system swap space
- Use data sampling for initial parameter optimization

```python
from clustering_framework import quick_cluster

# Use mini-batch processing
model, metrics = quick_cluster(
    data,
    algorithm="kmeans",
    batch_size=1000,  # Process 1000 samples at a time
    memory_efficient=True
)
```

#### 2. Slow Performance

**Problem**: Clustering takes too long to complete.

**Solutions**:
- Enable parallel processing
- Use approximate nearest neighbors for DBSCAN
- Reduce the parameter search space
- Use faster distance metrics

```python
# Enable parallel processing
results = optimize_clustering(
    data,
    algorithm="spectral",
    n_jobs=-1,  # Use all available cores
    approximate_neighbors=True  # Use approximate nearest neighbors
)
```

#### 3. Poor Clustering Results

**Problem**: Clusters don't match expected patterns.

**Solutions**:
- Try different preprocessing techniques
- Adjust algorithm parameters
- Use different distance metrics
- Check for data quality issues

```python
# Try different preprocessing
X = preprocess_data(
    data,
    scale=True,
    handle_missing='knn',  # Use KNN imputation
    feature_selection='variance'  # Remove low variance features
)
```

#### 4. Stability Issues

**Problem**: Clustering results are not stable across runs.

**Solutions**:
- Set random seed for reproducibility
- Increase number of initialization runs
- Use stability analysis to assess reliability
- Consider ensemble clustering

```python
# Ensure reproducibility
model, metrics = quick_cluster(
    data,
    random_state=42,
    n_init=20,  # More initialization runs
    ensemble=True  # Use ensemble clustering
)
```

### FAQ

#### Q: How do I choose the right clustering algorithm?

**A**: Consider these factors:
- Data size and dimensionality
- Expected cluster shapes
- Presence of noise
- Performance requirements

Quick guide:
- K-means: Spherical clusters, large datasets
- DBSCAN: Irregular shapes, automatic noise detection
- Spectral: Complex shapes, smaller datasets

#### Q: How do I determine the optimal number of clusters?

**A**: Use these methods:
1. Elbow method
2. Silhouette analysis
3. Gap statistic
4. Domain knowledge

```python
from clustering_framework.utils import find_optimal_clusters

n_clusters = find_optimal_clusters(
    data,
    max_clusters=10,
    methods=['elbow', 'silhouette', 'gap']
)
```

#### Q: How do I handle categorical variables?

**A**: Several approaches:
1. One-hot encoding for nominal variables
2. Label encoding for ordinal variables
3. Feature hashing for high-cardinality variables
4. Custom distance metrics

```python
# Handle mixed data types
X = preprocess_data(
    data,
    categorical_encoding='onehot',
    high_cardinality_method='hashing',
    n_components=20  # For feature hashing
)
```

#### Q: How do I validate clustering results?

**A**: Multiple validation strategies:
1. Internal validation metrics
2. Stability analysis
3. Cross-validation
4. Domain expert review

```python
# Comprehensive validation
validation = validate_clustering(
    data,
    model,
    internal_metrics=True,
    stability_analysis=True,
    cross_validation=True
)
```

#### Q: Can I use clustering for streaming data?

**A**: Yes, with these approaches:
1. Online clustering algorithms
2. Mini-batch processing
3. Incremental learning
4. Window-based clustering

```python
from clustering_framework import StreamingKMeans

# Setup streaming clustering
model = StreamingKMeans(
    n_clusters=5,
    window_size=1000,
    update_interval=100
)
```

### Getting Help

For additional support:
1. Check the [documentation](https://clustering-framework.readthedocs.io/)
2. Submit issues on [GitHub](https://github.com/username/clustering_framework/issues)
3. Join our [community forum](https://forum.clustering-framework.org)
4. Contact maintainers at support@clustering-framework.org
