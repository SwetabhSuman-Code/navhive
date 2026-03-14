# main.py (NavHive Phase 2)

import numpy as np
import matplotlib.pyplot as plt

from config import *
from models.anchor import Anchor
from models.hive_node import HiveNode
from engine.measurement import get_noisy_distances
from engine.trilateration import trilateration
from engine.aoa import get_noisy_aoa
from engine.adaptive_fusion import adaptive_fusion
from engine.anchor_health import check_anchor_health
from evaluation.metrics import compute_rmse


# =========================================================
# INITIALIZE NAVHIVE SYSTEM
# =========================================================

# Create shared anchors
anchors = [Anchor(x, y) for x, y in ANCHOR_POSITIONS]

# Create multiple blind nodes
NUM_NODES = 5
nodes = [HiveNode(i, [2+i, 3+i]) for i in range(NUM_NODES)]


# =========================================================
# MAIN SIMULATION LOOP
# =========================================================

for step in range(STEPS):

    for node in nodes:

        # Move node and get acceleration
        acceleration = node.move(step)

        # Store true position
        node.history.append(node.true_position.copy())

        # Measurements
        distances = get_noisy_distances(
            node.true_position,
            anchors,
            NOISE_DISTANCE
        )

        healthy_indices = check_anchor_health(distances)

        if len(healthy_indices) < 3:
            continue

        active_anchors = [anchors[i] for i in healthy_indices]
        active_distances = distances[healthy_indices]

        dist_est = trilateration(active_anchors, active_distances)

        aoa_est = get_noisy_aoa(
            node.true_position,
            anchors,
            NOISE_AOA
        )

        # EKF Prediction + Update using ONLY AOA
        node.kf.predict(acceleration)
        filtered_est = node.kf.update(aoa_est)

        node.filtered_history.append(filtered_est)
# =========================================================
# EVALUATION
# =========================================================

print("\n===== NAVHIVE PHASE 2 RESULTS =====")

for node in nodes:

    true_path = np.array(node.history)
    filtered_path = np.array(node.filtered_history)

    rmse = compute_rmse(filtered_path, true_path)
    print(f"Node {node.id} RMSE: {rmse:.4f}")


# =========================================================
# VISUALIZATION (MULTI-NODE)
# =========================================================

plt.figure(figsize=(10, 8))

# Plot anchors
anchor_positions = np.array([a.position for a in anchors])
plt.scatter(anchor_positions[:,0],
            anchor_positions[:,1],
            c='red',
            s=100,
            label="Anchors")

# Plot each node
for node in nodes:

    true_path = np.array(node.history)
    filtered_path = np.array(node.filtered_history)

    plt.plot(true_path[:,0],
             true_path[:,1],
             '--',
             label=f"Node {node.id} True")

    plt.plot(filtered_path[:,0],
             filtered_path[:,1],
             linewidth=2,
             label=f"Node {node.id} Filtered")

plt.title("NavHive Phase 2 - Multi Node Tracking")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()