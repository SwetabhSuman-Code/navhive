# config.py
# Global simulation parameters for NavHive
import numpy as np

# Four anchor positions forming a trapezoid layout (x, y) in metres
ANCHOR_POSITIONS = np.array([
    [0, 0],
    [10, 0],
    [5, 8],
    [10, 10]
])

NOISE_DISTANCE = 0.4   # Gaussian std-dev on distance measurements (metres)
NOISE_AOA      = 4     # Uniform noise on angle measurements (degrees)
DT             = 1     # Timestep duration (seconds)
STEPS          = 100   # Number of simulation iterations (increased for EKF convergence)