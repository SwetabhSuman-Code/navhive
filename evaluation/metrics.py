# evaluation/metrics.py
"""
Evaluation metrics for localization accuracy assessment.

Metrics:
    - RMSE (Root Mean Square Error): penalises large deviations more heavily.
    - MAE  (Mean Absolute Error): gives equal weight to all deviations.
"""

import numpy as np


def compute_rmse(estimates, true_path):
    """
    Compute Root Mean Square Error between estimated and true positions.

    Args:
        estimates  (np.ndarray): shape (N, 2) — EKF-filtered positions.
        true_path  (np.ndarray): shape (N, 2) — ground-truth positions.

    Returns:
        float: RMSE in the same units as the position coordinates.
    """
    return np.sqrt(np.mean(np.sum((estimates - true_path) ** 2, axis=1)))


def compute_mae(estimates, true_path):
    """
    Compute Mean Absolute Error between estimated and true positions.

    Args:
        estimates  (np.ndarray): shape (N, 2) — EKF-filtered positions.
        true_path  (np.ndarray): shape (N, 2) — ground-truth positions.

    Returns:
        float: MAE in the same units as the position coordinates.
    """
    return np.mean(np.linalg.norm(estimates - true_path, axis=1))