"""
tests/test_metrics.py

Unit tests for evaluation metrics (RMSE and MAE).

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from evaluation.metrics import compute_rmse, compute_mae


class TestRMSE:

    def test_zero_error(self):
        """Perfect estimates should give RMSE of 0."""
        estimates = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert compute_rmse(estimates, estimates) == pytest.approx(0.0, abs=1e-9)

    def test_known_rmse(self):
        """Manual RMSE: errors = [(1,0), (0,1)] → distances = [1, 1] → RMSE = 1."""
        estimates = np.array([[1.0, 0.0], [0.0, 1.0]])
        true_path = np.array([[0.0, 0.0], [0.0, 0.0]])
        # distances²: [1, 1] → mean = 1 → sqrt = 1
        assert compute_rmse(estimates, true_path) == pytest.approx(1.0, rel=1e-6)

    def test_rmse_positive(self):
        """RMSE must always be non-negative."""
        np.random.seed(7)
        estimates = np.random.randn(50, 2)
        true_path = np.random.randn(50, 2)
        assert compute_rmse(estimates, true_path) >= 0

    def test_rmse_penalises_outliers(self):
        """RMSE should be larger when one estimate is a large outlier."""
        good = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        bad  = np.array([[5.0, 5.0], [0.1, 0.1], [0.1, 0.1]])
        true = np.zeros((3, 2))
        assert compute_rmse(bad, true) > compute_rmse(good, true)


class TestMAE:

    def test_zero_error(self):
        """Perfect estimates should give MAE of 0."""
        estimates = np.array([[2.0, 3.0], [4.0, 5.0]])
        assert compute_mae(estimates, estimates) == pytest.approx(0.0, abs=1e-9)

    def test_known_mae(self):
        """Errors of distance 1 each → MAE = 1."""
        estimates = np.array([[1.0, 0.0], [0.0, 1.0]])
        true_path = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert compute_mae(estimates, true_path) == pytest.approx(1.0, rel=1e-6)

    def test_mae_positive(self):
        """MAE must always be non-negative."""
        np.random.seed(13)
        estimates = np.random.randn(30, 2)
        true_path = np.random.randn(30, 2)
        assert compute_mae(estimates, true_path) >= 0

    def test_mae_less_sensitive_to_outliers(self):
        """MAE should increase less than RMSE when an outlier is added."""
        true = np.zeros((4, 2))
        no_outlier   = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])
        with_outlier = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [10.0, 0.0]])
        from evaluation.metrics import compute_rmse
        rmse_delta = compute_rmse(with_outlier, true) - compute_rmse(no_outlier, true)
        mae_delta  = compute_mae(with_outlier, true)  - compute_mae(no_outlier, true)
        # RMSE delta should be larger — it penalises outliers more
        assert rmse_delta > mae_delta
