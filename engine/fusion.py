# engine/fusion.py
def hybrid_fusion(dist_est, aoa_est, weight=0.7):
    return weight * dist_est + (1-weight) * aoa_est