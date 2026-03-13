# models/blind_node.py
import numpy as np
from engine.kalman import DynamicKalman

class BlindNode:
    def __init__(self, init_pos):
        self.true_position = np.array(init_pos)
        self.kf = DynamicKalman()
        self.history = []

    def update_true_position(self, step):
        x = 2 + 0.12 * step
        y = 3 + 0.07 * step
        self.true_position = np.array([x, y])