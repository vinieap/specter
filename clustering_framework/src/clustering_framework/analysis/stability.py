"""
Stability analysis for clustering results.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score


def analyze_stability(
    X: np.ndarray,
    model: BaseEstimator,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Analyze clustering stability through cross-validation.
    
    Args:
        X: Input data matrix
        model: Fitted clustering model
        n_splits: Number of CV splits
        random_state: Random seed
        
    Returns:
        Dictionary with stability scores
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    stability_scores = []
    
    # Get base labels
    base_labels = model.predict(X)
    
    for train_idx, test_idx in kf.split(X):
        # Fit on training data
        model_cv = model.__class__(**model.get_params())
        model_cv.fit(X[train_idx])
        
        # Predict on test data
        test_labels = model_cv.predict(X[test_idx])
        base_test_labels = base_labels[test_idx]
        
        # Compare with base labels
        score = adjusted_rand_score(base_test_labels, test_labels)
        stability_scores.append(score)
    
    return {
        "mean_stability": np.mean(stability_scores),
        "std_stability": np.std(stability_scores),
        "individual_scores": stability_scores
    }