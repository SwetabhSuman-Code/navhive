# engine/aoa.py
"""
Angle of Arrival (AoA) estimation and ray intersection.

This module computes bearing angles from anchor stations to a target node,
injects measurement noise, and solves the resulting ray intersection geometry
to produce a 2-D position estimate.
"""

import numpy as np


def aoa(anchor, target):
    """
    Compute the true bearing angle (degrees) from an anchor to a target.

    Uses arctan2 so the angle is correctly resolved across all four quadrants.

    Args:
        anchor (array-like): (x, y) position of the anchor station.
        target (array-like): (x, y) position of the target node.

    Returns:
        float: Bearing angle in degrees.
    """
    return np.degrees(np.arctan2(target[1] - anchor[1],
                                 target[0] - anchor[0]))


def aoa_intersection(a1, t1, a2, t2):
    """
    Find the 2-D intersection point of two bearing rays.

    Each ray originates from an anchor position and travels at the given
    bearing angle. The intersection is solved via slope-intercept algebra.

    Args:
        a1 (array-like): (x, y) position of anchor 1.
        t1 (float):      Bearing angle (degrees) from anchor 1 to target.
        a2 (array-like): (x, y) position of anchor 2.
        t2 (float):      Bearing angle (degrees) from anchor 2 to target.

    Returns:
        np.ndarray: Estimated (x, y) position of the target.
    """
    m1 = np.tan(np.radians(t1))
    m2 = np.tan(np.radians(t2))

    x = (m1 * a1[1] - m2 * a2[1] + a2[0] - a1[0]) / (m1 - m2)
    y = m1 * (x - a1[0]) + a1[1]
    return np.array([x, y])


def get_noisy_aoa(true_pos, anchors, noise_deg):
    """
    Simulate noisy AoA measurement and return estimated position.

    Computes the true bearing from anchors 0 and 1 to the node, adds uniform
    angular noise, then calls aoa_intersection() to estimate the position.

    Args:
        true_pos  (array-like): Ground-truth (x, y) position of the node.
        anchors   (list):       List of Anchor objects with a .position attribute.
        noise_deg (float):      Half-width of the uniform noise range (degrees).
                                Noise is sampled from U(−noise_deg, +noise_deg).

    Returns:
        np.ndarray: AoA-based estimated (x, y) position.
    """
    t1 = aoa(anchors[0].position, true_pos) + np.random.uniform(-noise_deg, noise_deg)
    t2 = aoa(anchors[1].position, true_pos) + np.random.uniform(-noise_deg, noise_deg)
    return aoa_intersection(anchors[0].position, t1,
                            anchors[1].position, t2)