# engine/anchor_health.py
"""
Anchor Line-of-Sight (LOS) health checking.

Filters out anchors whose measured distance exceeds a physical plausibility
threshold, which indicates blocked or degraded line-of-sight (e.g. NLOS,
multipath, or hardware faults). Only healthy anchors participate in trilateration.
"""

import numpy as np


def check_anchor_health(distances, threshold=15):
    """
    Return indices of anchors with plausible (healthy) distance measurements.

    An anchor is considered healthy if its measured distance is below the
    given threshold. Distances above the threshold suggest a non-line-of-sight
    (NLOS) condition or hardware fault.

    Args:
        distances (np.ndarray): Measured distances from the node to each anchor.
        threshold (float):      Maximum acceptable distance (metres). Default = 15.

    Returns:
        list[int]: Indices of healthy anchors. The pipeline requires at least
                   3 healthy anchors for trilateration to proceed.
    """
    healthy = []
    for i, d in enumerate(distances):
        if d < threshold:
            healthy.append(i)
    return healthy