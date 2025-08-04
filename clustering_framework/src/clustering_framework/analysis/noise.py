"""
Noise sensitivity analysis for clustering results.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import adjusted_rand_score


def analyze_noise(
    X: np.ndarray,
    model: BaseEstimator,
    noise_levels: list[float] = [0.01, 0.05, 0.1],
    n_trials: int = 5,
    random_state: int = 42
) -> dict:
    """
    Analyze clustering sensitivity to noise.
    
    Args:
        X: Input data matrix
        model: Fitted clustering model
        noise_levels: List of noise standard deviations to test
        n_trials: Number of trials per noise level
        random_state: Random seed
        
    Returns:
        Dictionary with noise sensitivity scores
    """
    rng = np.random.RandomState(random_state)
    base_labels = model.predict(X)
    results = {}
    
    for noise_level in noise_levels:
        scores = []
        for _ in range(n_trials):
            # Add noise
            noise = rng.normal(0, noise_level, X.shape)
            X_noisy = X + noise
            
            # Fit and predict
            model_noisy = model.__class__(**model.get_params())
            model_noisy.fit(X_noisy)
            noisy_labels = model_noisy.predict(X_noisy)
            
            # Compare with base labels
            score = adjusted_rand_score(base_labels, noisy_labels)
            scores.append(score)
            
        results[f"noise_{noise_level}"] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "individual_scores": scores
        }
    
    return results