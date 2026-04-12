# engine/fusion.py
"""
Static-weight sensor fusion (baseline / legacy method).

Unlike adaptive_fusion.py which computes weights from noise levels,
this module uses a fixed user-defined weight. Useful as a comparison
baseline to demonstrate the advantage of adaptive weighting.
"""

import numpy as np


def hybrid_fusion(dist_est, aoa_est, weight=0.7):
    """
    Blend trilateration and AoA estimates with a fixed weight.

    Args:
        dist_est (np.ndarray): Position estimate from trilateration (x, y).
        aoa_est  (np.ndarray): Position estimate from AoA intersection (x, y).
        weight   (float): Weight assigned to the trilateration estimate.
                          (1 - weight) is given to AoA. Default = 0.7.

    Returns:
        np.ndarray: Blended position estimate (x, y).

    Note:
        For noise-adaptive weighting use adaptive_fusion() in adaptive_fusion.py.
    """
    return weight * np.array(dist_est) + (1 - weight) * np.array(aoa_est)