# Import all functionality from the modularized package
from spectral_clustering import *

# Re-export for backward compatibility
__all__ = [
    # Main API
    "optimize_spectral_clustering",
    # Optimizers
    "BatchBayesianSpectralOptimizer",
    "BayesianSpectralOptimizer",
    # Configuration
    "VerbosityLevel",
    "DEFAULT_VERBOSITY",
    "N_CORES",
    "PARAM_NAMES",
    # Utilities
    "get_memory_usage",
    "get_array_size_mb",
    "format_params_for_display",
    # Parameters
    "sample_params",
    "prepare_clusterer_params",
    # Evaluation
    "evaluate_params_worker",
    "parallel_objective_function",
    # Visualization
    "generate_optimization_visualizations",
    "create_optimization_summary_plots",
]


if __name__ == "__main__":
    # Run the example performance comparison
    from spectral_clustering.examples import run_performance_comparison

    run_performance_comparison()
