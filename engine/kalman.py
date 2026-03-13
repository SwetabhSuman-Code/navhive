# engine/kalman.py (Phase 3 - IMU Based EKF)

import numpy as np

class DynamicKalman:
    def __init__(self, dt=1):

        self.dt = dt

        # State: [x, y, vx, vy]
        self.x = np.zeros(4)

        self.P = np.eye(4) * 500

        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.5

    def predict(self, acceleration):

        ax, ay = acceleration
        dt = self.dt

        # State transition with acceleration
        F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ])

        B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])

        self.x = F @ self.x + B @ np.array([ax, ay])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):

        H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.x[:2]