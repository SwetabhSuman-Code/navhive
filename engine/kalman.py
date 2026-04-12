# engine/kalman.py
"""
Dynamic Kalman Filter (Extended Kalman Filter — EKF) for 2-D position tracking.

State vector: [x, y, vx, vy]  (position + velocity in both axes)

The filter operates in two phases per timestep:
    1. Predict — propagates the state forward using constant-acceleration kinematics.
    2. Update  — corrects the prediction with a noisy 2-D position measurement
                 (from adaptive sensor fusion) using the standard Kalman gain equation.

Matrices:
    F  — 4×4 state-transition matrix (constant-velocity model)
    B  — 4×2 control-input matrix    (maps [ax, ay] acceleration to state)
    H  — 2×4 observation matrix      (extracts x, y from [x, y, vx, vy])
    Q  — 4×4 process noise covariance
    R  — 2×2 measurement noise covariance
    P  — 4×4 state covariance (uncertainty)
"""

import numpy as np


class DynamicKalman:
    """
    A 4-state Extended Kalman Filter for 2-D position and velocity tracking.

    Tracks a single node using position measurements and an optional
    acceleration input (from a simulated IMU / motion model).
    """

    def __init__(self, dt=1):
        """
        Initialise the Kalman Filter with default matrices.

        Args:
            dt (float): Timestep duration in seconds. Default = 1.
        """
        self.dt = dt

        # State: [x, y, vx, vy] — initialised at origin with zero velocity
        self.x = np.zeros(4)

        # Large initial covariance = high uncertainty before any measurements
        self.P = np.eye(4) * 500

        self.Q = np.eye(4) * 0.01   # Process noise (model imperfection)
        self.R = np.eye(2) * 0.5    # Measurement noise covariance

    def predict(self, acceleration):
        """
        Predict the next state using constant-acceleration kinematics.

        Propagates the state estimate forward one timestep using:
            x_k = F * x_{k-1} + B * [ax, ay]
            P_k = F * P_{k-1} * F^T + Q

        Args:
            acceleration (array-like): [ax, ay] — acceleration input at this step.
                                       Pass [0, 0] for static nodes.
        """
        ax, ay = acceleration
        dt = self.dt

        # State-transition matrix (constant-velocity kinematics)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Control-input matrix (maps acceleration to position and velocity)
        B = np.array([
            [0.5 * dt ** 2, 0],
            [0, 0.5 * dt ** 2],
            [dt, 0],
            [0,  dt]
        ])

        self.x = F @ self.x + B @ np.array([ax, ay])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """
        Correct the state estimate using a 2-D position measurement.

        Applies the standard Kalman update equations:
            y = z - H * x_k        (innovation / residual)
            S = H * P * H^T + R    (innovation covariance)
            K = P * H^T * S^{-1}   (Kalman gain)
            x = x + K * y
            P = (I - K * H) * P

        Args:
            measurement (np.ndarray): Observed (x, y) position from sensor fusion.

        Returns:
            np.ndarray: Filtered (x, y) position estimate.
        """
        # Observation matrix — extracts [x, y] from the 4-D state
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.x[:2]