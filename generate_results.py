"""
generate_results.py — Pre-generate all NavHive result plots

Runs the full simulation and saves all output figures to the results/ folder
as high-resolution PNGs. This allows evaluators to view results without
running the code themselves.

Generates:
    results/01_trajectory_all_nodes.png   — True vs EKF paths for all nodes
    results/02_noise_vs_rmse.png          — RMSE as a function of sensor noise
    results/03_comparison_stages.png      — Stage-by-stage algorithm comparison

Usage:
    python generate_results.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt

from config import ANCHOR_POSITIONS, NOISE_DISTANCE, NOISE_AOA, STEPS
from models.anchor import Anchor
from models.hive_node import HiveNode
from engine.measurement import get_noisy_distances
from engine.trilateration import trilateration
from engine.aoa import get_noisy_aoa
from engine.adaptive_fusion import adaptive_fusion
from engine.anchor_health import check_anchor_health
from evaluation.metrics import compute_rmse, compute_mae

os.makedirs("results", exist_ok=True)
print("Generating results...")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Trajectory: True path vs EKF-filtered path for all nodes
# ══════════════════════════════════════════════════════════════════════════════

NUM_NODES = 5
anchors = [Anchor(x, y) for x, y in ANCHOR_POSITIONS]
nodes   = [HiveNode(i, [2 + i, 3 + i]) for i in range(NUM_NODES)]

for step in range(STEPS):
    for node in nodes:
        acceleration = node.move(step)
        node.history.append(node.true_position.copy())

        distances       = get_noisy_distances(node.true_position, anchors, NOISE_DISTANCE)
        healthy_indices = check_anchor_health(distances)
        if len(healthy_indices) < 3:
            continue

        active_anchors   = [anchors[i] for i in healthy_indices]
        active_distances = distances[healthy_indices]

        dist_est   = trilateration(active_anchors, active_distances)
        aoa_est    = get_noisy_aoa(node.true_position, anchors, NOISE_AOA)
        hybrid_est = adaptive_fusion(dist_est, aoa_est, NOISE_DISTANCE, NOISE_AOA)

        node.kf.predict(acceleration)
        filtered = node.kf.update(hybrid_est)
        node.filtered_history.append(filtered.copy())

fig, ax = plt.subplots(figsize=(10, 8))
anchor_pos = np.array([a.position for a in anchors])
ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
           c="red", s=120, marker="^", zorder=5, label="Anchors")

for node in nodes:
    true_arr     = np.array(node.history)
    filtered_arr = np.array(node.filtered_history)
    if len(filtered_arr) == 0:
        continue
    rmse = compute_rmse(filtered_arr, true_arr[:len(filtered_arr)])
    mae  = compute_mae(filtered_arr,  true_arr[:len(filtered_arr)])
    ax.plot(true_arr[:, 0],     true_arr[:, 1],     "--",  alpha=0.6,
            label=f"Node {node.id} True")
    ax.plot(filtered_arr[:, 0], filtered_arr[:, 1], linewidth=2,
            label=f"Node {node.id} EKF  RMSE={rmse:.3f} MAE={mae:.3f}")

ax.set_title("NavHive — Multi-Node Static Localization (EKF Filtered)", fontsize=13)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/01_trajectory_all_nodes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> results/01_trajectory_all_nodes.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Noise vs. RMSE curve
# ══════════════════════════════════════════════════════════════════════════════

from simulator import NavHiveSimulator

noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
rmse_list    = []
mae_list     = []

for noise in noise_levels:
    sim    = NavHiveSimulator(num_nodes=5)
    errors = []
    for step in range(60):
        positions = sim.step(step)
        for p in positions:
            errors.append(np.linalg.norm(p["true"] - p["filtered"]))
    rmse_list.append(np.sqrt(np.mean(np.array(errors) ** 2)))
    mae_list.append(np.mean(errors))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(noise_levels, rmse_list, "o-", linewidth=2, label="RMSE", color="#E74C3C")
ax.plot(noise_levels, mae_list,  "s--", linewidth=2, label="MAE",  color="#3498DB")
ax.set_title("Sensor Noise vs. Localization Error", fontsize=13)
ax.set_xlabel("Distance Sensor Noise σ (m)")
ax.set_ylabel("Error (m)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/02_noise_vs_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> results/02_noise_vs_rmse.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Algorithm stage comparison (calls comparison_plot logic)
# ══════════════════════════════════════════════════════════════════════════════

anchors = [Anchor(x, y) for x, y in ANCHOR_POSITIONS]
node    = HiveNode(0, [3.0, 4.0])
tri_h, aoa_h, ekf_h = [], [], []

for step in range(STEPS):
    acceleration     = node.move(step)
    distances        = get_noisy_distances(node.true_position, anchors, NOISE_DISTANCE)
    healthy          = check_anchor_health(distances)
    if len(healthy) < 3:
        continue
    active_anchors   = [anchors[i] for i in healthy]
    active_distances = distances[healthy]

    tri_est  = trilateration(active_anchors, active_distances)
    aoa_est  = get_noisy_aoa(node.true_position, anchors, NOISE_AOA)
    hybrid   = adaptive_fusion(tri_est, aoa_est, NOISE_DISTANCE, NOISE_AOA)
    node.kf.predict(acceleration)
    ekf_est  = node.kf.update(hybrid)

    tri_h.append(tri_est)
    aoa_h.append(aoa_est)
    ekf_h.append(ekf_est.copy())

tri_arr = np.array(tri_h)
aoa_arr = np.array(aoa_h)
ekf_arr = np.array(ekf_h)
true_pos = node.true_position

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("NavHive — Algorithm Stage Comparison (Static Node)", fontsize=13, fontweight="bold")

for ax, arr, label, color in [
    (axes[0], tri_arr, "Stage 1: Trilateration Only",    "#E74C3C"),
    (axes[1], aoa_arr, "Stage 2: AoA Only",              "#9B59B6"),
    (axes[2], ekf_arr, "Stage 3: Adaptive Fusion + EKF", "#2980B9"),
]:
    ax.scatter(arr[:, 0], arr[:, 1], c=color, alpha=0.45, s=18, label="Estimates")
    ax.scatter(*true_pos, c="#2ECC71", s=150, zorder=5, marker="*", label="True Pos")
    for a in anchors:
        ax.scatter(*a.position, c="#F39C12", s=80, marker="^", zorder=4)
    rmse = np.sqrt(np.mean(np.sum((arr - true_pos) ** 2, axis=1)))
    mae  = np.mean(np.linalg.norm(arr - true_pos, axis=1))
    ax.set_title(f"{label}\nRMSE={rmse:.3f} m  |  MAE={mae:.3f} m", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Y (m)")
plt.tight_layout()
plt.savefig("results/03_comparison_stages.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> results/03_comparison_stages.png")

print("\nAll results saved to results/")
