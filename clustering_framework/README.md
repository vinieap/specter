# Clustering Framework

A Python framework focused on noise detection and analysis in clustering tasks. The framework provides advanced tools for identifying and characterizing different types of noise in your data, helping you make informed decisions about data preprocessing and algorithm selection.

## Overview

The framework is built to address key challenges in handling noisy data for clustering:
- Detecting and characterizing different types of noise points
- Identifying outliers using multiple statistical methods
- Analyzing local and global noise patterns
- Providing actionable recommendations for noise handling

### Key Features
- **Advanced Noise Detection**: Multiple methods including statistical, density-based, and isolation-based approaches
- **Noise Classification**: Identification of different noise types (global outliers, local outliers, density-based noise)
- **Ensemble Approach**: Combines multiple detection methods for robust noise identification
- **Actionable Insights**: Provides specific recommendations for algorithm selection and parameter tuning
- **Comprehensive Analysis**: Detailed noise profiles with multiple metrics and scores

### When to Use This Framework
- **Data Preprocessing**: Identify and handle noise before clustering
- **Algorithm Selection**: Get recommendations for noise-robust clustering methods
- **Parameter Tuning**: Receive data-driven suggestions for clustering parameters
- **Quality Assessment**: Understand the noise characteristics of your dataset

## Features

### Noise Detection Methods

#### Statistical Outlier Detection
- **Approach**: Uses statistical methods to identify outliers
- **Key Features**:
  - Z-score analysis for each feature
  - Mahalanobis distance calculation
  - Combined statistical evidence
- **Parameters**:
  - Contamination level
  - Statistical thresholds
- **Output**:
  - Noise scores
  - Statistical metrics
  - Outlier indices

#### Density-Based Detection
- **Approach**: Identifies noise based on local density patterns
- **Key Features**:
  - K-nearest neighbors density estimation
  - Reachability analysis
  - Local density ratio calculation
- **Parameters**:
  - Number of neighbors
  - Density thresholds
- **Output**:
  - Density scores
  - Reachability metrics
  - Density-based noise indices

#### Isolation Forest Detection
- **Approach**: Uses isolation trees to find anomalies
- **Key Features**:
  - Random forest-based isolation
  - Path length analysis
  - Anomaly scoring
- **Parameters**:
  - Contamination level
  - Random state
- **Output**:
  - Isolation scores
  - Anomaly predictions
  - Forest statistics

#### Local Outlier Detection
- **Approach**: Identifies local outliers using LOF
- **Key Features**:
  - Local Outlier Factor calculation
  - Neighborhood analysis
  - Local density comparison
- **Parameters**:
  - Number of neighbors
  - Contamination level
- **Output**:
  - LOF scores
  - Outlier factors
  - Local outlier indices

#### Covariance-Based Detection
- **Approach**: Uses robust covariance estimation
- **Key Features**:
  - Elliptic envelope fitting
  - Robust covariance estimation
  - Mahalanobis scoring
- **Parameters**:
  - Contamination level
  - Random state
- **Output**:
  - Covariance scores
  - Location estimates
  - Precision matrix

### Noise Analysis Tools

#### Noise Classification
- **Types Detected**:
  - Global outliers
  - Local outliers
  - Density-based noise
  - Bridge points
- **Features**:
  - Multi-criteria classification
  - Hierarchical noise typing
  - Confidence scoring

#### Recommendations Engine
- **Algorithm Suggestions**:
  - Based on noise ratio
  - Considers noise types
  - Specific to data characteristics
- **Parameter Recommendations**:
  - DBSCAN parameters
  - Spectral clustering parameters
  - Preprocessing suggestions
- **Preprocessing Advice**:
  - Based on noise patterns
  - Data-specific guidance
  - Handling strategies

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

### Dependencies

The framework requires the following packages:
- numpy>=1.20.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- optuna>=3.0.0
- optuna-dashboard>=0.12.0 (optional: for visualization)

### Installation

```bash
# Create and activate virtual environment
python -m venv clustering_env
source clustering_env/bin/activate  # On Windows: clustering_env\Scripts\activate

# Clone the repository
git clone https://github.com/username/clustering_framework.git
cd clustering_framework

# Install in development mode
pip install -e .

## Usage Examples

Here are examples demonstrating the framework's noise detection and analysis capabilities:

### Basic Noise Detection

```python
import numpy as np
from clustering_framework import NoiseDetector

# Create sample data
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
X[0:50] += 10  # Add some obvious outliers

# Initialize noise detector
detector = NoiseDetector(
    contamination=0.1,  # Expected proportion of noise
    n_neighbors=20,     # For local noise detection
    random_state=42     # For reproducibility
)

# Detect and analyze noise
noise_profile = detector.detect_noise(X)

# Print basic statistics
print(f"Noise ratio: {noise_profile.noise_ratio:.3f}")
print(f"Number of noise points: {len(noise_profile.noise_indices)}")

# Examine different types of noise
for noise_type, indices in noise_profile.noise_types.items():
    print(f"\n{noise_type}: {sum(indices)} points")

# Get recommendations
print("\nRecommended algorithms:", noise_profile.recommendations["algorithm"])
print("\nPreprocessing suggestions:", noise_profile.recommendations["preprocessing"])
```

### Detailed Noise Analysis

```python
import pandas as pd
from clustering_framework import NoiseDetector

# Load your dataset
data = pd.read_csv('your_dataset.csv')
X = data.values

# Initialize detector with custom settings
detector = NoiseDetector(
    contamination=0.15,  # Higher contamination expectation
    n_neighbors=30       # More neighbors for stable density estimation
)

# Get noise profile
profile = detector.detect_noise(X)

# Examine noise scores
print("\nNoise Score Statistics:")
print(f"Mean score: {np.mean(profile.noise_scores):.3f}")
print(f"Score threshold: {np.percentile(profile.noise_scores, 85):.3f}")

# Look at different noise types
print("\nNoise Type Breakdown:")
for noise_type, indices in profile.noise_types.items():
    print(f"{noise_type}: {sum(indices)} points ({sum(indices)/len(X)*100:.1f}%)")

# Get algorithm-specific recommendations
if "DBSCAN" in profile.recommendations["algorithm"]:
    eps = profile.recommendations["parameters"]["eps"]
    min_samples = profile.recommendations["parameters"]["min_samples"]
    print(f"\nRecommended DBSCAN parameters:")
    print(f"eps: {eps:.3f}")
    print(f"min_samples: {min_samples}")

# Check preprocessing recommendations
print("\nPreprocessing Recommendations:")
for rec in profile.recommendations["preprocessing"]:
    print(f"- {rec}")
```

### Accessing Detailed Analysis Results

```python
# Get detailed results from each detection method
statistical_results = profile.details["_detect_statistical_outliers"]
density_results = profile.details["_detect_density_based_noise"]
isolation_results = profile.details["_detect_isolation_forest_noise"]

# Examine statistical outliers
print("\nStatistical Analysis:")
print(f"Z-score range: {np.min(statistical_results['z_scores'])} to {np.max(statistical_results['z_scores'])}")
print(f"Mahalanobis distance threshold: {statistical_results['threshold']:.3f}")

# Look at density-based results
print("\nDensity Analysis:")
print(f"Average density score: {np.mean(density_results['density_scores']):.3f}")
print(f"Reachability range: {np.min(density_results['reachability'])} to {np.max(density_results['reachability'])}")

# Check isolation forest insights
print("\nIsolation Forest Results:")
print(f"Number of estimators used: {isolation_results['n_estimators']}")
```

Each example demonstrates:
- How to configure the noise detector
- Different ways to analyze noise in your data
- How to interpret the results
- How to use the recommendations
```

## API Reference

### NoiseDetector Class

```python
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
        """
        Initialize noise detector.

        Parameters
        ----------
        contamination : float, default=0.1
            Expected proportion of noise points
        n_neighbors : int, default=20
            Number of neighbors for local noise detection
        random_state : int, default=42
            Random state for reproducible results
        """

    def detect_noise(self, X: np.ndarray) -> NoiseProfile:
        """
        Detect and analyze noise in the data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)

        Returns
        -------
        NoiseProfile
            Detailed noise analysis results
        """
```

### NoiseProfile Class

```python
@dataclass
class NoiseProfile:
    """
    Profile of detected noise in the data.
    
    Attributes
    ----------
    noise_ratio : float
        Ratio of noise points to total points
    noise_indices : np.ndarray
        Indices of noise points
    noise_types : Dict[str, np.ndarray]
        Different types of noise points
    noise_scores : np.ndarray
        Noise scores for each point
    recommendations : Dict[str, Any]
        Recommendations for handling noise
    details : Dict[str, Any]
        Detailed analysis results
    """
```

### Detection Methods

The NoiseDetector class includes five different detection methods:

1. **Statistical Outlier Detection**
   ```python
   _detect_statistical_outliers(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
   ```
   - Uses Z-scores and Mahalanobis distances
   - Returns noise mask, scores, and statistical details

2. **Density-Based Detection**
   ```python
   _detect_density_based_noise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
   ```
   - Uses k-nearest neighbors density estimation
   - Returns noise mask, scores, and density metrics

3. **Isolation Forest Detection**
   ```python
   _detect_isolation_forest_noise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
   ```
   - Uses isolation forest algorithm
   - Returns noise mask, scores, and forest details

4. **Local Outlier Detection**
   ```python
   _detect_local_outliers(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
   ```
   - Uses Local Outlier Factor (LOF)
   - Returns noise mask, scores, and LOF details

5. **Covariance-Based Detection**
   ```python
   _detect_covariance_based_noise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
   ```
   - Uses robust covariance estimation
   - Returns noise mask, scores, and covariance details

### Analysis Methods

1. **Noise Classification**
   ```python
   _classify_noise_types(
       X: np.ndarray,
       noise_scores: np.ndarray,
       method_results: Dict[str, Any]
   ) -> Dict[str, np.ndarray]
   ```
   - Classifies noise into different types
   - Returns masks for each noise type

2. **Recommendations Generation**
   ```python
   _generate_recommendations(
       X: np.ndarray,
       noise_ratio: float,
       noise_types: Dict[str, np.ndarray],
       method_results: Dict[str, Any]
   ) -> Dict[str, Any]
   ```
   - Generates algorithm and preprocessing recommendations
   - Returns detailed suggestions based on noise analysis

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
