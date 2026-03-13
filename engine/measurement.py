# engine/measurement.py
import numpy as np

def get_noisy_distances(true_pos, anchors, noise_std):
    anchor_positions = np.array([a.position for a in anchors])
    distances = np.linalg.norm(anchor_positions - true_pos, axis=1)
    noise = np.random.normal(0, noise_std, len(distances))
    return distances + noise