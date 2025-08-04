Welcome to Clustering Library Documentation
=====================================

A comprehensive library for clustering optimization and analysis.

Features
--------

- Bayesian optimization of clustering parameters
- Multiple clustering algorithms support
- Advanced analysis tools
- Easy-to-use convenience functions
- Comprehensive validation and error handling

Quick Start
----------

Installation
~~~~~~~~~~~

.. code-block:: bash

   pip install clustering

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from clustering import quick_cluster
   from sklearn.datasets import make_blobs

   # Generate sample data
   X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

   # Perform clustering with automatic parameter selection
   model, metrics = quick_cluster(X)
   print(f"Silhouette score: {metrics['silhouette']:.3f}")

Advanced Usage
~~~~~~~~~~~~

.. code-block:: python

   from clustering import optimize_clustering, analyze_and_improve

   # Optimize clustering parameters
   results = optimize_clustering(
       X,
       algorithm="dbscan",
       n_calls=100,
       use_dashboard=True
   )

   # Analyze and improve results
   improved_model, analysis = analyze_and_improve(
       X,
       results.best_model
   )

   # Print recommendations
   for rec in analysis['recommendations']:
       print(f"- {rec}")

Contents
--------

.. toctree::
   :maxdepth: 2

   api
   examples
   tutorials
   architecture

Indices and Tables
----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`