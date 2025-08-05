# Type System Documentation

This document describes the type system used in the clustering framework and provides guidelines for using types effectively.

## Core Types

### Type Aliases

The framework provides several type aliases for commonly used types:

```python
from clustering_framework.core.types import ParamDict, FloatArray, IntArray, ArrayLike

# Dictionary of parameters
params: ParamDict = {"n_clusters": 3, "tol": 0.001}

# Numpy array of float64
data: FloatArray = np.array([[1.0, 2.0], [3.0, 4.0]])

# Numpy array of int64
labels: IntArray = np.array([0, 1, 0, 2])

# Any array-like input
inputs: ArrayLike = [[1.0, 2.0], [3.0, 4.0]]
```

### Generic Types

The framework uses generic types to provide better type safety:

```python
from clustering_framework.core.types import EstimatorT, ClusteringEstimatorT

class MyEstimator(BaseEstimator):
    def fit(self, X: ArrayLike) -> EstimatorT:
        ...

class MyClusterer(BaseEstimator, ClusterMixin):
    def fit_predict(self, X: ArrayLike) -> IntArray:
        ...
```

### Protocols

The framework defines several protocols that describe common interfaces:

```python
from clustering_framework.core.types import HasFit, HasPredict, HasFitPredict, ClusteringEstimator

# Classes can implement these protocols
class MyAlgorithm(HasFitPredict[MyAlgorithm]):
    def fit(self, X: ArrayLike) -> MyAlgorithm:
        ...
    def predict(self, X: ArrayLike) -> IntArray:
        ...
```

## Runtime Type Checking

The framework provides decorators for runtime type checking:

```python
from clustering_framework.core.decorators import type_check, validate_params

@type_check
def process_data(X: ArrayLike, n_clusters: int) -> FloatArray:
    ...

@validate_params(n_clusters=int, tol=float)
def optimize_clusters(n_clusters: int, tol: float = 0.001) -> None:
    ...
```

## Result Types

The framework includes several dataclasses for structured results:

```python
from clustering_framework.core.types import (
    OptimizationResult,
    ValidationResult,
    NoiseAnalysis,
    AlgorithmPerformance
)

def optimize() -> OptimizationResult:
    ...

def validate(params: ParamDict) -> ValidationResult:
    ...

def analyze_noise(X: ArrayLike) -> NoiseAnalysis:
    ...

def benchmark(algorithm: str) -> AlgorithmPerformance:
    ...
```

## Configuration Types

The framework uses dataclasses for configuration:

```python
from clustering_framework.core.types import AlgorithmConfig

config = AlgorithmConfig(
    name="kmeans",
    category="partitioning",
    params={"n_clusters": 3},
    param_ranges={"tol": (1e-5, 1e-3)},
    param_choices={"init": ["k-means++", "random"]},
    param_dependencies={}
)
```

## Best Practices

1. Always use type hints for function arguments and return values
2. Use the provided type aliases instead of raw types
3. Use protocols instead of concrete classes when possible
4. Add runtime type checking for critical functions
5. Use validation decorators for parameter validation
6. Document any custom types or type variables
7. Use mypy for static type checking

## Type Checking

The framework uses mypy for static type checking with strict settings:

```ini
# mypy.ini
[mypy]
python_version = 3.8
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
```

Run type checking with:

```bash
mypy src/clustering_framework
```