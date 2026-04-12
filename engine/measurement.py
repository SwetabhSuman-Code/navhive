# engine/measurement.py
"""
Sensor measurement simulation — noisy distance generation.

Computes true Euclidean distances from a node to each anchor, then
injects independent Gaussian noise to simulate real RSSI/ToF hardware.
"""

import numpy as np


def get_noisy_distances(true_pos, anchors, noise_std):
    """
    Compute noisy distance measurements from a node to all anchors.

    Each true distance is corrupted by independent Gaussian noise,
    simulating real-world RSSI or Time-of-Flight measurement errors.

    Args:
        true_pos  (array-like): Ground-truth (x, y) position of the node.
        anchors   (list):       List of Anchor objects (each has a .position attribute).
        noise_std (float):      Standard deviation of the Gaussian noise (metres).

    Returns:
        np.ndarray: Array of noisy distances, one per anchor.
    """
    anchor_positions = np.array([a.position for a in anchors])
    distances = np.linalg.norm(anchor_positions - true_pos, axis=1)
    noise = np.random.normal(0, noise_std, len(distances))
    return distances + noise