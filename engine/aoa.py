# engine/aoa.py
import numpy as np

def aoa(anchor, target):
    return np.degrees(np.arctan2(target[1]-anchor[1],
                                 target[0]-anchor[0]))

def aoa_intersection(a1, t1, a2, t2):
    m1 = np.tan(np.radians(t1))
    m2 = np.tan(np.radians(t2))

    x = (m1*a1[1] - m2*a2[1] + a2[0] - a1[0]) / (m1 - m2)
    y = m1*(x - a1[0]) + a1[1]
    return np.array([x, y])

def get_noisy_aoa(true_pos, anchors, noise_deg):
    t1 = aoa(anchors[0].position, true_pos) + np.random.uniform(-noise_deg, noise_deg)
    t2 = aoa(anchors[1].position, true_pos) + np.random.uniform(-noise_deg, noise_deg)
    return aoa_intersection(anchors[0].position, t1,
                            anchors[1].position, t2)