# models/hive_node.py (Phase 3)

import numpy as np
from engine.kalman import DynamicKalman

class HiveNode:
    def __init__(self, node_id, init_pos):
        self.id = node_id
        self.true_position = np.array(init_pos, dtype=float)
        # Static node — no velocity
        self.kf = DynamicKalman()
        self.history = []
        self.filtered_history = []

    def move(self, step):
        # Static node — position does not change
        return np.array([0.0, 0.0])