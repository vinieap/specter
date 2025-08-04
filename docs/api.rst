API Reference
============

This section provides detailed documentation for all public APIs in the clustering library.

Core API
--------

.. module:: clustering.api

These functions provide the core functionality for clustering optimization and analysis.

optimize_clustering
~~~~~~~~~~~~~~~~~

.. autofunction:: optimize_clustering

analyze_clusters
~~~~~~~~~~~~~~

.. autofunction:: analyze_clusters

evaluate_clustering
~~~~~~~~~~~~~~~~~

.. autofunction:: evaluate_clustering

compare_algorithms
~~~~~~~~~~~~~~~~

.. autofunction:: compare_algorithms

Convenience Functions
-------------------

.. module:: clustering.convenience

These functions provide simplified interfaces for common clustering operations.

quick_cluster
~~~~~~~~~~~~

.. autofunction:: quick_cluster

analyze_and_improve
~~~~~~~~~~~~~~~~~

.. autofunction:: analyze_and_improve

find_optimal_clusters
~~~~~~~~~~~~~~~~~~~

.. autofunction:: find_optimal_clusters

compare_and_recommend
~~~~~~~~~~~~~~~~~~~

.. autofunction:: compare_and_recommend

Core Types
---------

.. module:: clustering.core.types

These classes define the core data structures used throughout the library.

OptimizationResult
~~~~~~~~~~~~~~~~

.. autoclass:: OptimizationResult
   :members:
   :undoc-members:
   :show-inheritance:

NoiseAnalysis
~~~~~~~~~~~~

.. autoclass:: NoiseAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

AlgorithmPerformance
~~~~~~~~~~~~~~~~~~

.. autoclass:: AlgorithmPerformance
   :members:
   :undoc-members:
   :show-inheritance:

Deprecated APIs
-------------

.. module:: clustering.compat

These functions are maintained for backwards compatibility and will be removed in a future version.

.. warning::
   These functions are deprecated. Please use the new API functions instead.

optimize_spectral_clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: optimize_spectral_clustering

optimize_kmeans
~~~~~~~~~~~~~

.. autofunction:: optimize_kmeans

analyze_noise
~~~~~~~~~~~

.. autofunction:: analyze_noise

analyze_convergence
~~~~~~~~~~~~~~~~~

.. autofunction:: analyze_convergence