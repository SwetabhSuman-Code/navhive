import numpy as np

from config import *
from models.anchor import Anchor
from models.hive_node import HiveNode
from engine.measurement import get_noisy_distances
from engine.trilateration import trilateration
from engine.aoa import get_noisy_aoa
from engine.adaptive_fusion import adaptive_fusion
from engine.anchor_health import check_anchor_health


class NavHiveSimulator:

    def __init__(self, num_nodes=5):

        # create anchors
        self.anchors = [Anchor(x, y) for x, y in ANCHOR_POSITIONS]

        # create nodes
        self.nodes = [HiveNode(i, [2 + i, 3 + i]) for i in range(num_nodes)]

        # store anchor history for visualization
        self.anchor_history = [[] for _ in self.anchors]


    def step(self, step):

        positions = []

        # ----- move anchors -----
        for i, anchor in enumerate(self.anchors):
            anchor.move(step)
            self.anchor_history[i].append(anchor.position.copy())

        # ----- process nodes -----
        for node in self.nodes:

            acceleration = node.move(step)

            distances = get_noisy_distances(
                node.true_position,
                self.anchors,
                NOISE_DISTANCE
            )

            healthy_indices = check_anchor_health(distances)

            if len(healthy_indices) < 3:
                continue

            active_anchors = [self.anchors[i] for i in healthy_indices]
            active_distances = distances[healthy_indices]

            dist_est = trilateration(active_anchors, active_distances)

            aoa_est = get_noisy_aoa(
                node.true_position,
                self.anchors,
                NOISE_AOA
            )

            hybrid_est = adaptive_fusion(
                dist_est,
                aoa_est,
                NOISE_DISTANCE,
                NOISE_AOA
            )

            # EKF prediction and update
            node.kf.predict(acceleration)
            filtered_est = node.kf.update(hybrid_est)

            node.history.append(node.true_position.copy())
            node.filtered_history.append(filtered_est)

            positions.append({
                "true": node.true_position,
                "filtered": filtered_est
            })

        return positions