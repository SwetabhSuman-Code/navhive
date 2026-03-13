# models/hive_node.py (Phase 3)

import numpy as np
from engine.kalman import DynamicKalman

class HiveNode:
    def __init__(self, node_id, init_pos):
        self.id = node_id
        self.true_position = np.array(init_pos, dtype=float)
        self.velocity = np.array([0.1, 0.05], dtype=float)
        self.kf = DynamicKalman()
        self.history = []
        self.filtered_history = []

    def move(self, step):

        # Simulated acceleration pattern
        ax = 0.003 * np.sin(0.1 * step)
        ay = 0.003 * np.cos(0.1 * step)

        acceleration = np.array([ax, ay])

        # Update true motion
        self.velocity += acceleration
        self.true_position += self.velocity

        return acceleration