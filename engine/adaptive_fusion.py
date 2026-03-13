# engine/adaptive_fusion.py
import numpy as np

def adaptive_fusion(dist_est, aoa_est, dist_noise, aoa_noise):

    # Lower noise = higher weight
    w_dist = 1 / (dist_noise + 1e-6)
    w_aoa = 1 / (aoa_noise + 1e-6)

    total = w_dist + w_aoa
    w_dist /= total
    w_aoa /= total

    return w_dist * dist_est + w_aoa * aoa_est