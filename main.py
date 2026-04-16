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
from evaluation.metrics import compute_rmse, compute_mae


# =========================================================
# INITIALIZE NAVHIVE SYSTEM
# =========================================================

# Create shared anchors
anchors = [Anchor(x, y) for x, y in ANCHOR_POSITIONS]

# Create multiple blind nodes
NUM_NODES = 5
nodes = [HiveNode(i, [2+i, 3+i]) for i in range(NUM_NODES)]


import csv

# Open CSV for logging raw inputs as requested by the professor
raw_log_file = open("raw_sensor_inputs.csv", "w", newline="")
raw_csv_writer = csv.writer(raw_log_file)
# Write header for the raw inputs log
raw_csv_writer.writerow(["Step", "NodeID", "True_X", "True_Y", "Dist_A0", "Dist_A1", "Dist_A2", "Dist_A3"])

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================

for step in range(STEPS):

    for node in nodes:

        # Move node and get acceleration
        acceleration = node.move(step)

        # Store true position
        node.history.append(node.true_position.copy())

        # Measurements - these are the REAL INPUTS (raw sensor data)
        distances = get_noisy_distances(
            node.true_position,
            anchors,
            NOISE_DISTANCE
        )

        # Log the real, unmanipulated inputs to the CSV
        log_row = [
            step, 
            node.id, 
            f"{node.true_position[0]:.4f}", 
            f"{node.true_position[1]:.4f}"
        ] + [f"{d:.4f}" for d in distances]
        raw_csv_writer.writerow(log_row)

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

raw_log_file.close()
# =========================================================
# EVALUATION
# =========================================================

results = []
for node in nodes:
    true_path     = np.array(node.history)
    filtered_path = np.array(node.filtered_history)
    if len(filtered_path) == 0:
        continue
    rmse = compute_rmse(filtered_path, true_path[:len(filtered_path)])
    mae  = compute_mae(filtered_path,  true_path[:len(filtered_path)])
    pos  = node.true_position
    pred_pos = np.mean(filtered_path, axis=0) # Average filtered position across simulation
    quality = "Excellent" if rmse < 1.0 else "Good" if rmse < 2.5 else "Fair" if rmse < 5.0 else "Poor"
    results.append((node.id, pos[0], pos[1], pred_pos[0], pred_pos[1], rmse, mae, quality))

# --- Print bullet-point results ---
print()
print("=" * 50)
print("  NavHive -- Localization Evaluation Results")
print("=" * 50)
print()
print("  [Setup]")
print("   * Simulation Steps : {}".format(STEPS))
print("   * Number of Nodes  : {}".format(NUM_NODES))
print("   * Distance Noise   : {} m (Gaussian std-dev)".format(NOISE_DISTANCE))
print("   * AoA Noise        : {} deg (Uniform)".format(NOISE_AOA))
print()
print("  [Node Results]")
print()

for nid, x, y, px, py, rmse, mae, qual in results:
    print("   Node {}".format(nid))
    print("    * Actual Position    : x={:.2f} m, y={:.2f} m".format(x, y))
    print("    * Predicted Position : x={:.2f} m, y={:.2f} m".format(px, py))
    print("    * RMSE               : {:.4f} m".format(rmse))
    print("    * MAE                : {:.4f} m".format(mae))
    print("    * Quality            : {}".format(qual))
    print()

avg_rmse = np.mean([r[5] for r in results])
avg_mae  = np.mean([r[6] for r in results])
best     = min(results, key=lambda r: r[5])
worst    = max(results, key=lambda r: r[5])

print("-" * 50)
print("  [Summary]")
print("   * Average RMSE : {:.4f} m".format(avg_rmse))
print("   * Average MAE  : {:.4f} m".format(avg_mae))
print("   * Best Node    : Node {} (RMSE = {:.4f} m)".format(best[0], best[5]))
print("   * Worst Node   : Node {} (RMSE = {:.4f} m)".format(worst[0], worst[5]))
print("=" * 50)
print()


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