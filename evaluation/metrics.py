# evaluation/metrics.py
import numpy as np

def compute_rmse(estimates, true_path):
    return np.sqrt(np.mean(np.sum((estimates - true_path)**2, axis=1)))