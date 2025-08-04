Examples
========

This section provides detailed examples of using the clustering library for various tasks.

Basic Clustering
--------------

Quick Start
~~~~~~~~~~

The simplest way to perform clustering is using the :func:`quick_cluster` function:

.. code-block:: python

   from clustering import quick_cluster
   from sklearn.datasets import make_blobs
   import numpy as np

   # Generate sample data
   X, _ = make_blobs(
       n_samples=300,
       centers=4,
       cluster_std=0.60,
       random_state=42
   )

   # Perform clustering
   model, metrics = quick_cluster(X, n_clusters=4)

   # Print results
   print(f"Silhouette score: {metrics['silhouette']:.3f}")
   print(f"Calinski-Harabasz score: {metrics['calinski_harabasz']:.3f}")

Optimizing Parameters
-------------------

Basic Optimization
~~~~~~~~~~~~~~~~

Use :func:`optimize_clustering` for fine-grained control over the optimization process:

.. code-block:: python

   from clustering import optimize_clustering

   # Optimize KMeans clustering
   results = optimize_clustering(
       X,
       algorithm="kmeans",
       n_calls=100,
       use_dashboard=True,
       dashboard_port=8080
   )

   print(f"Best score: {results.best_score:.3f}")
   print("Best parameters:")
   for param, value in results.best_params.items():
       print(f"  {param}: {value}")

Comparing Algorithms
~~~~~~~~~~~~~~~~~~

Compare multiple algorithms to find the best one:

.. code-block:: python

   from clustering import compare_and_recommend

   # Compare algorithms
   best_model, comparison = compare_and_recommend(
       X,
       algorithms=['kmeans', 'dbscan', 'spectral'],
       n_clusters=4
   )

   print(f"Best algorithm: {comparison['best_algorithm']}")
   print("\\nRecommendations:")
   for rec in comparison['recommendations']:
       print(f"- {rec}")

Analysis and Improvement
----------------------

Basic Analysis
~~~~~~~~~~~~

Analyze clustering results:

.. code-block:: python

   from clustering import analyze_clusters

   # Analyze clustering
   analysis = analyze_clusters(
       X,
       model,
       noise_analysis=True,
       stability_analysis=True
   )

   # Print noise analysis
   if "noise_analysis" in analysis:
       print(f"Noise ratio: {analysis['noise_analysis'].noise_ratio:.2%}")
       print("Recommendations:")
       for rec in analysis['noise_analysis'].recommendations:
           print(f"- {rec}")

   # Print stability analysis
   if "stability_scores" in analysis:
       print(f"\\nStability: {analysis['stability_scores']['mean_stability']:.2%}")

Automatic Improvement
~~~~~~~~~~~~~~~~~~

Use :func:`analyze_and_improve` to automatically improve clustering results:

.. code-block:: python

   from clustering import analyze_and_improve

   # Analyze and improve
   improved_model, analysis = analyze_and_improve(
       X,
       model,
       improve=True
   )

   print("Recommendations:")
   for rec in analysis['recommendations']:
       print(f"- {rec}")

Finding Optimal Clusters
----------------------

Automatic Detection
~~~~~~~~~~~~~~~~~

Use :func:`find_optimal_clusters` to automatically determine the optimal number of clusters:

.. code-block:: python

   from clustering import find_optimal_clusters

   # Find optimal number of clusters
   n_clusters = find_optimal_clusters(
       X,
       max_clusters=15,
       algorithm='kmeans'
   )

   print(f"Optimal number of clusters: {n_clusters}")

Manual Comparison
~~~~~~~~~~~~~~~

Compare different numbers of clusters:

.. code-block:: python

   from clustering import optimize_clustering, evaluate_clustering
   import numpy as np

   scores = []
   for n in range(2, 11):
       # Optimize clustering
       results = optimize_clustering(
           X,
           algorithm='kmeans',
           n_clusters=n,
           n_calls=50
       )
       
       # Evaluate results
       metrics = evaluate_clustering(
           X,
           results.best_model
       )
       
       scores.append({
           'n_clusters': n,
           'silhouette': metrics['silhouette'],
           'calinski_harabasz': metrics['calinski_harabasz']
       })

   # Find best n
   best_score = max(scores, key=lambda x: x['silhouette'])
   print(f"Best n_clusters: {best_score['n_clusters']}")
   print(f"Silhouette score: {best_score['silhouette']:.3f}")

Working with Large Datasets
-------------------------

Batch Processing
~~~~~~~~~~~~~~

Use batch optimization for large datasets:

.. code-block:: python

   from clustering import optimize_clustering

   # Generate large dataset
   X = np.random.randn(10000, 10)

   # Use batch optimization
   results = optimize_clustering(
       X,
       algorithm='mini_batch_kmeans',
       n_calls=100,
       batch_size=8,
       n_jobs=-1,  # Use all CPU cores
       use_batch_optimizer=True
   )

Memory Efficiency
~~~~~~~~~~~~~~~

Handle memory-constrained situations:

.. code-block:: python

   from clustering import optimize_clustering
   import numpy as np

   # Generate data in chunks
   chunk_size = 1000
   n_chunks = 10
   results = []

   for i in range(n_chunks):
       # Generate chunk
       X_chunk = np.random.randn(chunk_size, 10)
       
       # Optimize clustering
       chunk_results = optimize_clustering(
           X_chunk,
           algorithm='mini_batch_kmeans',
           n_calls=30
       )
       
       results.append(chunk_results)

   # Aggregate results
   best_score = max(r.best_score for r in results)
   print(f"Best overall score: {best_score:.3f}")

Real-time Monitoring
------------------

Using the Dashboard
~~~~~~~~~~~~~~~~~

Monitor optimization progress in real-time:

.. code-block:: python

   from clustering import optimize_clustering

   # Enable dashboard
   results = optimize_clustering(
       X,
       algorithm='kmeans',
       n_calls=100,
       use_dashboard=True,
       dashboard_port=8080
   )

   # Dashboard will be available at http://localhost:8080

Custom Progress Tracking
~~~~~~~~~~~~~~~~~~~~~

Track optimization progress manually:

.. code-block:: python

   from clustering import optimize_clustering
   from tqdm import tqdm

   # Create progress bar
   with tqdm(total=100, desc="Optimizing") as pbar:
       def progress_callback(study, trial):
           pbar.update(1)
       
       # Run optimization
       results = optimize_clustering(
           X,
           algorithm='kmeans',
           n_calls=100,
           optimizer_kwargs={'callbacks': [progress_callback]}
       )