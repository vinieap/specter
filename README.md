# Specter

A Python library for optimizing spectral clustering parameters using Bayesian optimization with Optuna.

## Overview

Specter provides two optimization approaches for tuning spectral clustering hyperparameters:

- **Batch Optimizer**: Parallel evaluation using ask-and-tell interface for high performance
- **Sequential Optimizer**: Traditional sequential Bayesian optimization

Both optimizers support real-time visualization through Optuna dashboard integration.

## Installation

The project uses Python with dependencies on scikit-learn, optuna, and related packages.

## Usage

### Basic Example

```python
from spectral_clustering import optimize_spectral_clustering
from sklearn.datasets import make_blobs

# Generate or load your data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Optimize spectral clustering parameters
results = optimize_spectral_clustering(
    X, 
    n_calls=50,
    use_batch_optimizer=True,
    use_dashboard=True
)

print(f"Best score: {results['best_score']}")
print(f"Best parameters: {results['best_params']}")
```

### Direct Optimizer Usage

```python
from spectral_clustering import BatchBayesianSpectralOptimizer

optimizer = BatchBayesianSpectralOptimizer(
    n_calls=100,
    batch_size=8,
    use_dashboard=True
)
optimizer.set_data(X)
results = optimizer.optimize()
```

## Features

- Bayesian optimization of spectral clustering hyperparameters
- Parallel batch evaluation for improved performance
- Real-time optimization monitoring via web dashboard
- Automatic visualization generation
- Memory usage tracking and optimization
- Support for custom parameter ranges and constraints

## Project Structure

```
spectral_clustering/
├── api.py              # Main optimization interface
├── batch_optimizer.py  # Batch Bayesian optimizer
├── sequential_optimizer.py  # Sequential optimizer
├── config.py          # Configuration and constants
├── parameters.py      # Parameter sampling and preparation
├── evaluation.py      # Objective function and evaluation
├── utils.py          # Utility functions
├── visualization.py  # Plot generation
└── examples.py       # Usage examples
```

## Testing

Run the test script to verify functionality:

```bash
python test_optuna_spectral.py
```

This script demonstrates both optimization approaches and generates comparison visualizations.

## Output

The optimizers generate:
- Parameter importance plots
- Optimization history plots
- Parallel coordinate plots
- Performance comparison visualizations
- Optuna study database for persistent storage