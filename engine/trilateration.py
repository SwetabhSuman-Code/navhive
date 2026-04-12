# engine/trilateration.py
"""
Distance-based trilateration using least-squares optimisation.

Solves the overdetermined system of circle equations to find the most
likely 2-D position of a node given noisy distances from ≥ 3 anchors.

Method:
    The squared-distance equation for anchor i:
        (x - xi)^2 + (y - yi)^2 = di^2
    Subtracting anchor 0's equation from each other anchor linearises the
    system into Ax = b form, which is solved with numpy.linalg.lstsq.

    Row i of A : [2(xi - x0), 2(yi - y0)]
    Row i of b : d0^2 - di^2 - x0^2 + xi^2 - y0^2 + yi^2
"""

import numpy as np


def trilateration(anchors, distances):
    """
    Estimate a node's 2-D position from distances to multiple anchors.

    Args:
        anchors   (list):       List of Anchor objects (each has a .position attribute).
                                Must contain at least 3 anchors.
        distances (np.ndarray): Measured (possibly noisy) distances from the node
                                to each anchor, in the same order as `anchors`.

    Returns:
        np.ndarray: Estimated (x, y) position of the node.
    """
    anchor_positions = np.array([a.position for a in anchors])
    A = []
    B = []

    x1, y1 = anchor_positions[0]
    d1 = distances[0]

    for i in range(1, len(anchor_positions)):
        xi, yi = anchor_positions[i]
        di = distances[i]

        A.append([2 * (xi - x1), 2 * (yi - y1)])
        B.append(d1 ** 2 - di ** 2 - x1 ** 2 + xi ** 2 - y1 ** 2 + yi ** 2)

    A = np.array(A)
    B = np.array(B)

    pos = np.linalg.lstsq(A, B, rcond=None)[0]
    return pos