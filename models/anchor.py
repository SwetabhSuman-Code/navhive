import numpy as np

class Anchor:

    def __init__(self, x, y):
        self.position = np.array([x, y], dtype=float)

    def move(self, step):
        # visible circular motion
        dx = 0.15 * np.sin(0.05 * step)
        dy = 0.15 * np.cos(0.05 * step)

        self.position += np.array([dx, dy])