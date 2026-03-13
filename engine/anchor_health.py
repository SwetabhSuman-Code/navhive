# engine/anchor_health.py
import numpy as np

def check_anchor_health(distances, threshold=15):
    healthy = []
    for i, d in enumerate(distances):
        if d < threshold:  # unrealistic distance means faulty
            healthy.append(i)
    return healthy