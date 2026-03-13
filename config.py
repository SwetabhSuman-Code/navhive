# config.py
import numpy as np

ANCHOR_POSITIONS = np.array([
    [0, 0],
    [10, 0],
    [5, 8],
    [10, 10]
])

NOISE_DISTANCE = 0.4
NOISE_AOA = 4
DT = 1
STEPS = 60