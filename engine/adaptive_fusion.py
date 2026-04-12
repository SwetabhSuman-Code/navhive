# engine/adaptive_fusion.py
"""
Adaptive inverse-noise sensor fusion.

Blends two independent position estimates (from trilateration and AoA) by
weighting each estimate inversely proportional to its sensor noise level.
A lower noise level → higher confidence → higher weight.

Weight formula:
    w_dist = (1/σ_dist) / (1/σ_dist + 1/σ_aoa)
    w_aoa  = 1 - w_dist
    p_fused = w_dist * p_tri + w_aoa * p_aoa

With default config (NOISE_DISTANCE=0.4, NOISE_AOA=4):
    w_dist ≈ 0.909  →  trilateration dominates (~91%).
"""

import numpy as np


def adaptive_fusion(dist_est, aoa_est, dist_noise, aoa_noise):
    """
    Fuse trilateration and AoA estimates with inverse-noise weighting.

    Args:
        dist_est  (np.ndarray): Position estimate from trilateration (x, y).
        aoa_est   (np.ndarray): Position estimate from AoA intersection (x, y).
        dist_noise (float):     Gaussian std-dev of the distance measurements (metres).
        aoa_noise  (float):     Noise magnitude of the AoA measurements (degrees).

    Returns:
        np.ndarray: Fused position estimate (x, y).
    """
    # Lower noise → higher weight (small epsilon avoids division by zero)
    w_dist = 1 / (dist_noise + 1e-6)
    w_aoa  = 1 / (aoa_noise  + 1e-6)

    total   = w_dist + w_aoa
    w_dist /= total
    w_aoa  /= total

    return w_dist * dist_est + w_aoa * aoa_est