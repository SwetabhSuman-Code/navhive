# engine/trilateration.py
import numpy as np

def trilateration(anchors, distances):
    anchor_positions = np.array([a.position for a in anchors])
    A = []
    B = []

    x1, y1 = anchor_positions[0]
    d1 = distances[0]

    for i in range(1, len(anchor_positions)):
        xi, yi = anchor_positions[i]
        di = distances[i]

        A.append([2*(xi - x1), 2*(yi - y1)])
        B.append(d1**2 - di**2 - x1**2 + xi**2 - y1**2 + yi**2)

    A = np.array(A)
    B = np.array(B)

    pos = np.linalg.lstsq(A, B, rcond=None)[0]
    return pos