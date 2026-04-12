"""
comparison_plot.py — NavHive Algorithm Stage Comparison

Runs 100 simulation steps for a single static node and plots the estimated
position at each step for three pipeline stages:

    Stage 1 — Trilateration only     (distance-based, no filtering)
    Stage 2 — AoA only               (angle-based, no filtering)
    Stage 3 — Adaptive Fusion + EKF  (full pipeline, our best estimate)

This plot clearly demonstrates the incremental improvement each stage adds,
and is the key visualisation for the project report.

Usage:
    python comparison_plot.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import ANCHOR_POSITIONS, NOISE_DISTANCE, NOISE_AOA, STEPS
from models.anchor import Anchor
from models.hive_node import HiveNode
from engine.measurement import get_noisy_distances
from engine.trilateration import trilateration
from engine.aoa import get_noisy_aoa
from engine.adaptive_fusion import adaptive_fusion
from engine.anchor_health import check_anchor_health

# ── Setup ─────────────────────────────────────────────────────────────────────

anchors = [Anchor(x, y) for x, y in ANCHOR_POSITIONS]
node    = HiveNode(0, [3.0, 4.0])   # single static node at (3, 4)

tri_history    = []
aoa_history    = []
fused_history  = []
ekf_history    = []

# ── Simulation Loop ───────────────────────────────────────────────────────────

for step in range(STEPS):

    acceleration = node.move(step)   # returns [0, 0] — node is static

    distances = get_noisy_distances(node.true_position, anchors, NOISE_DISTANCE)
    healthy   = check_anchor_health(distances)

    if len(healthy) < 3:
        continue

    active_anchors   = [anchors[i] for i in healthy]
    active_distances = distances[healthy]

    # Stage 1 — Trilateration only
    tri_est = trilateration(active_anchors, active_distances)
    tri_history.append(tri_est)

    # Stage 2 — AoA only
    aoa_est = get_noisy_aoa(node.true_position, anchors, NOISE_AOA)
    aoa_history.append(aoa_est)

    # Stage 3 — Adaptive Fusion → EKF
    hybrid_est   = adaptive_fusion(tri_est, aoa_est, NOISE_DISTANCE, NOISE_AOA)
    node.kf.predict(acceleration)
    filtered_est = node.kf.update(hybrid_est)
    ekf_history.append(filtered_est.copy())

# ── Conversion ────────────────────────────────────────────────────────────────

tri_arr = np.array(tri_history)
aoa_arr = np.array(aoa_history)
ekf_arr = np.array(ekf_history)

true_pos = node.true_position   # stays fixed (static node)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("NavHive — Algorithm Stage Comparison (Static Node)", fontsize=14, fontweight="bold")

COLORS = {
    "true":  "#2ECC71",
    "tri":   "#E74C3C",
    "aoa":   "#9B59B6",
    "ekf":   "#2980B9",
    "anchor":"#F39C12",
}

for ax, estimates, label, color in [
    (axes[0], tri_arr, "Stage 1: Trilateration Only",    COLORS["tri"]),
    (axes[1], aoa_arr, "Stage 2: AoA Only",              COLORS["aoa"]),
    (axes[2], ekf_arr, "Stage 3: Adaptive Fusion + EKF", COLORS["ekf"]),
]:
    # Scatter of estimates
    ax.scatter(estimates[:, 0], estimates[:, 1],
               c=color, alpha=0.45, s=18, label="Estimates")

    # True position marker
    ax.scatter(*true_pos, c=COLORS["true"], s=150, zorder=5,
               marker="*", label="True Position")

    # Anchor positions
    for a in anchors:
        ax.scatter(*a.position, c=COLORS["anchor"], s=80, marker="^", zorder=4)

    # Centroid of estimates
    centroid = estimates.mean(axis=0)
    ax.scatter(*centroid, c=color, s=100, marker="X", edgecolors="black",
               linewidths=0.8, zorder=6, label=f"Mean Est.")

    rmse = np.sqrt(np.mean(np.sum((estimates - true_pos) ** 2, axis=1)))
    mae  = np.mean(np.linalg.norm(estimates - true_pos, axis=1))

    ax.set_title(f"{label}\nRMSE={rmse:.3f} m  |  MAE={mae:.3f} m", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Y (m)")

# Legend for anchors
anchor_patch = mpatches.Patch(color=COLORS["anchor"], label="Anchor")
fig.legend(handles=[anchor_patch], loc="lower right", fontsize=9)

plt.tight_layout()

# Save to results/ folder
os.makedirs("results", exist_ok=True)
plt.savefig("results/comparison_plot.png", dpi=150, bbox_inches="tight")
print("Saved → results/comparison_plot.png")
plt.show()
