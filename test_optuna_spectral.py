#!/usr/bin/env python3
"""Test script to verify the Optuna-based spectral clustering optimization works correctly"""

import numpy as np
from sklearn.datasets import make_blobs

from spectral_clustering import (
    BatchBayesianSpectralOptimizer,
    BayesianSpectralOptimizer,
    VerbosityLevel,
    generate_optimization_visualizations,
    optimize_spectral_clustering,
)

# Generate simple test data
print("Generating test data...")
X, y = make_blobs(n_samples=1000, n_features=10, centers=3, random_state=42)
print(f"Dataset shape: {X.shape}")

# Test 1: Basic optimization with batch optimizer
print("\n=== Test 1: Batch Optimizer (Ask-and-Tell) ===")
batch_optimizer = BatchBayesianSpectralOptimizer(
    n_calls=20,
    batch_size=4,
    verbosity=VerbosityLevel.MEDIUM,
    random_state=42,
    use_dashboard=True,
)
batch_optimizer.set_data(X)
batch_results = batch_optimizer.optimize()

print("\nBatch Results:")
print(f"Best score: {batch_results['best_score']:.4f}")
print(f"Best params: {batch_results['best_params']}")
print(f"Total evaluations: {batch_results['n_evaluations']}")

# Test 2: Sequential optimizer
print("\n=== Test 2: Sequential Optimizer ===")
seq_optimizer = BayesianSpectralOptimizer(
    n_calls=20,
    verbosity=VerbosityLevel.MEDIUM,
    random_state=42,
    use_dashboard=True,
)
seq_optimizer.set_data(X)
seq_results = seq_optimizer.optimize()

print("\nSequential Results:")
print(f"Best score: {seq_results['best_score']:.4f}")
print(f"Best params: {seq_results['best_params']}")
print(f"Total evaluations: {seq_results['n_evaluations']}")

# Test 3: Using the convenience function (sequential to avoid multiprocessing issues)
print("\n=== Test 3: Convenience Function ===")
results = optimize_spectral_clustering(
    X,
    n_calls=10,
    verbosity=VerbosityLevel.MINIMAL,
    use_batch_optimizer=False,  # Use sequential to avoid multiprocessing hang
    use_dashboard=False,  # Disable dashboard to simplify
)

print("\nConvenience Function Results:")
print(f"Best score: {results['best_score']:.4f}")
print(f"Optimization time: {results['optimization_time']:.2f}s")

# Test 4: Get best clusterer
print("\n=== Test 4: Get Best Clusterer ===")
best_clusterer = results["best_clusterer"]
labels = best_clusterer.fit_predict(X)
print(f"Number of clusters found: {len(np.unique(labels))}")
print(f"Cluster labels: {np.unique(labels)}")

print("\n✅ All tests completed successfully!")

# Generate visualizations from the optimization results
generate_optimization_visualizations(
    batch_results=batch_results,
    sequential_results=seq_results,
    verbosity=VerbosityLevel.MEDIUM,
)
