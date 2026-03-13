# hive_manager.py
import numpy as np
from models.hive_node import HiveNode
from models.anchor import Anchor
from config import *
from engine.measurement import get_noisy_distances
from engine.trilateration import trilateration
from engine.aoa import get_noisy_aoa
from engine.adaptive_fusion import adaptive_fusion
from engine.anchor_health import check_anchor_health
from evaluation.metrics import compute_rmse

class NavHive:

    def __init__(self, num_nodes=3):
        self.anchors = [Anchor(x,y) for x,y in ANCHOR_POSITIONS]
        self.nodes = [HiveNode(i, [2+i, 3+i]) for i in range(num_nodes)]

    def run(self):

        for step in range(STEPS):

            for node in self.nodes:

                node.move(step)

                distances = get_noisy_distances(node.true_position,
                                                self.anchors,
                                                NOISE_DISTANCE)

                healthy_indices = check_anchor_health(distances)

                active_anchors = [self.anchors[i] for i in healthy_indices]
                active_distances = distances[healthy_indices]

                if len(active_anchors) < 3:
                    continue

                dist_est = trilateration(active_anchors, active_distances)
                aoa_est = get_noisy_aoa(node.true_position, self.anchors, NOISE_AOA)

                hybrid_est = adaptive_fusion(dist_est,
                                             aoa_est,
                                             NOISE_DISTANCE,
                                             NOISE_AOA)

                filtered_est = node.kf.update(hybrid_est)

                node.history.append(node.true_position)
                node.filtered_history.append(filtered_est)

        self.evaluate()

    def evaluate(self):
        for node in self.nodes:
            true_path = np.array(node.history)
            filtered_path = np.array(node.filtered_history)

            rmse = compute_rmse(filtered_path, true_path)
            print(f"Node {node.id} RMSE:", rmse)