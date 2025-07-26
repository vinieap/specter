from .batch_optimizer import BatchBayesianSpectralOptimizer
from .config import DEFAULT_VERBOSITY
from .sequential_optimizer import BayesianSpectralOptimizer


def optimize_spectral_clustering(
    X,
    n_calls=100,
    n_jobs=None,
    batch_size=None,
    verbosity=DEFAULT_VERBOSITY,
    use_batch_optimizer=True,
    use_dashboard=False,
    dashboard_port=8080,
    **optimizer_kwargs,
):
    """
    Optimize spectral clustering parameters for a dataset using Bayesian optimization.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input dataset to cluster
    n_calls : int, default=100
        Number of optimization iterations
    n_jobs : int, default=None
        Number of parallel jobs (defaults to cpu_count() - 1)
    batch_size : int, default=None
        Batch size for parallel evaluation (defaults to n_jobs * 2, max 8)
    verbosity : int, default=VerbosityLevel.DETAILED
        Verbosity level for output
    use_batch_optimizer : bool, default=True
        Whether to use the high-performance BatchBayesianSpectralOptimizer (recommended)
    use_dashboard : bool, default=False
        Whether to start optuna-dashboard for real-time visualization
    dashboard_port : int, default=8080
        Port for the optuna-dashboard web interface
    **optimizer_kwargs : dict
        Additional arguments passed to the optimizer

    Returns:
    --------
    dict : Optimization results including best parameters and clusterer
    """
    if use_batch_optimizer:
        optimizer = BatchBayesianSpectralOptimizer(
            n_calls=n_calls,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbosity=verbosity,
            use_dashboard=use_dashboard,
            dashboard_port=dashboard_port,
            **optimizer_kwargs,
        )
    else:
        optimizer = BayesianSpectralOptimizer(
            n_calls=n_calls,
            n_jobs=n_jobs,
            verbosity=verbosity,
            use_dashboard=use_dashboard,
            dashboard_port=dashboard_port,
        )

    optimizer.set_data(X)
    results = optimizer.optimize()

    # Add the best clusterer to results
    results["best_clusterer"] = optimizer.get_best_clusterer()

    return results
