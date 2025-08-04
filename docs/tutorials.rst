Tutorials
=========

This section provides step-by-step tutorials for common clustering workflows.

Basic Clustering Workflow
-----------------------

This tutorial walks through the basic process of clustering data.

1. Preparing Your Data
~~~~~~~~~~~~~~~~~~~~

First, let's prepare some sample data:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_blobs
   from sklearn.preprocessing import StandardScaler

   # Generate sample data
   X, true_labels = make_blobs(
       n_samples=300,
       centers=4,
       cluster_std=0.60,
       random_state=42
   )

   # Scale the data
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

2. Finding the Optimal Number of Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, let's determine how many clusters to use:

.. code-block:: python

   from clustering import find_optimal_clusters

   # Find optimal number of clusters
   n_clusters = find_optimal_clusters(
       X_scaled,
       max_clusters=10,
       algorithm='kmeans'
   )

   print(f"Optimal number of clusters: {n_clusters}")

3. Optimizing Clustering Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can optimize the clustering parameters:

.. code-block:: python

   from clustering import optimize_clustering

   # Optimize clustering
   results = optimize_clustering(
       X_scaled,
       algorithm='kmeans',
       n_calls=100,
       n_clusters=n_clusters,
       use_dashboard=True
   )

4. Analyzing Results
~~~~~~~~~~~~~~~~~

Let's analyze the clustering results:

.. code-block:: python

   from clustering import analyze_clusters

   # Analyze clustering
   analysis = analyze_clusters(
       X_scaled,
       results.best_model,
       noise_analysis=True,
       stability_analysis=True
   )

   # Print analysis results
   if 'noise_analysis' in analysis:
       print(f"Noise ratio: {analysis['noise_analysis'].noise_ratio:.2%}")
       for rec in analysis['noise_analysis'].recommendations:
           print(f"- {rec}")

5. Visualizing Results
~~~~~~~~~~~~~~~~~~~

Finally, let's visualize the results:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Get cluster labels
   labels = results.best_model.labels_

   # Create scatter plot
   plt.figure(figsize=(10, 6))
   plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
   plt.title('Clustering Results')
   plt.show()

Advanced Optimization Tutorial
---------------------------

This tutorial covers advanced optimization techniques.

1. Setting Up Multiple Studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's set up multiple optimization studies:

.. code-block:: python

   from clustering import compare_algorithms

   # Compare multiple algorithms
   results = compare_algorithms(
       X_scaled,
       algorithms=['kmeans', 'dbscan', 'spectral'],
       n_calls=50,
       n_jobs=-1
   )

2. Customizing Parameter Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom parameter ranges for optimization:

.. code-block:: python

   from clustering import optimize_clustering

   # Custom parameter ranges
   optimizer_kwargs = {
       'param_ranges': {
           'n_clusters': (2, 10),
           'max_iter': (100, 500),
           'tol': (1e-5, 1e-3, 'log')
       }
   }

   # Run optimization
   results = optimize_clustering(
       X_scaled,
       algorithm='kmeans',
       n_calls=100,
       **optimizer_kwargs
   )

3. Using Callbacks
~~~~~~~~~~~~~~~

Implement custom callbacks for monitoring:

.. code-block:: python

   def progress_callback(study, trial):
       score = trial.value
       params = trial.params
       print(f"Trial {trial.number}:")
       print(f"  Score: {score:.3f}")
       print("  Parameters:", params)

   # Run optimization with callback
   results = optimize_clustering(
       X_scaled,
       algorithm='kmeans',
       n_calls=50,
       optimizer_kwargs={'callbacks': [progress_callback]}
   )

4. Parallel Processing
~~~~~~~~~~~~~~~~~~~

Optimize performance with parallel processing:

.. code-block:: python

   # Enable parallel processing
   results = optimize_clustering(
       X_scaled,
       algorithm='kmeans',
       n_calls=100,
       n_jobs=-1,
       batch_size=8,
       use_batch_optimizer=True
   )

Working with Large Datasets
-------------------------

This tutorial shows how to handle large datasets efficiently.

1. Using Mini-Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use mini-batch algorithms for large data:

.. code-block:: python

   from clustering import optimize_clustering
   import numpy as np

   # Generate large dataset
   X_large = np.random.randn(100000, 10)

   # Use mini-batch optimization
   results = optimize_clustering(
       X_large,
       algorithm='mini_batch_kmeans',
       n_calls=50,
       batch_size=1000,
       n_jobs=-1
   )

2. Implementing Data Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~

Process data in streams:

.. code-block:: python

   from clustering import quick_cluster
   import numpy as np

   def data_generator(n_chunks, chunk_size):
       for _ in range(n_chunks):
           yield np.random.randn(chunk_size, 10)

   # Process data in chunks
   models = []
   for chunk in data_generator(10, 1000):
       model, metrics = quick_cluster(chunk)
       models.append((model, metrics))

3. Memory-Efficient Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze large datasets without memory issues:

.. code-block:: python

   from clustering import analyze_clusters
   import numpy as np

   def analyze_in_chunks(X, model, chunk_size=1000):
       results = []
       for i in range(0, len(X), chunk_size):
           chunk = X[i:i + chunk_size]
           chunk_analysis = analyze_clusters(
               chunk,
               model,
               noise_analysis=True
           )
           results.append(chunk_analysis)
       return results

Real-time Monitoring Tutorial
--------------------------

This tutorial covers real-time monitoring of clustering optimization.

1. Setting Up the Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~

Enable and configure the dashboard:

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

2. Custom Progress Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

Implement custom progress tracking:

.. code-block:: python

   from clustering import optimize_clustering
   from tqdm import tqdm
   import time

   class ProgressTracker:
       def __init__(self, n_trials):
           self.pbar = tqdm(total=n_trials)
           self.start_time = time.time()
           self.best_score = float('-inf')
       
       def __call__(self, study, trial):
           self.pbar.update(1)
           if trial.value > self.best_score:
               self.best_score = trial.value
               elapsed = time.time() - self.start_time
               print(f"\\nNew best score: {trial.value:.3f}")
               print(f"Time elapsed: {elapsed:.1f}s")
               print("Parameters:", trial.params)

   # Use progress tracker
   tracker = ProgressTracker(n_trials=100)
   results = optimize_clustering(
       X,
       algorithm='kmeans',
       n_calls=100,
       optimizer_kwargs={'callbacks': [tracker]}
   )

3. Saving Progress
~~~~~~~~~~~~~~~

Save optimization progress for later analysis:

.. code-block:: python

   import json
   from pathlib import Path

   class ProgressSaver:
       def __init__(self, save_dir):
           self.save_dir = Path(save_dir)
           self.save_dir.mkdir(exist_ok=True)
           self.history = []
       
       def __call__(self, study, trial):
           result = {
               'number': trial.number,
               'value': trial.value,
               'params': trial.params,
               'datetime': trial.datetime.isoformat()
           }
           self.history.append(result)
           
           # Save after every 10 trials
           if len(self.history) % 10 == 0:
               with open(self.save_dir / 'progress.json', 'w') as f:
                   json.dump(self.history, f, indent=2)

   # Use progress saver
   saver = ProgressSaver('optimization_results')
   results = optimize_clustering(
       X,
       algorithm='kmeans',
       n_calls=100,
       optimizer_kwargs={'callbacks': [saver]}
   )