# visualization/plotter.py
import matplotlib.pyplot as plt
import numpy as np

def plot_results(anchors, true_path, raw, hybrid, filtered):

    anchor_positions = np.array([a.position for a in anchors])

    plt.figure(figsize=(10,8))
    plt.scatter(anchor_positions[:,0], anchor_positions[:,1], c='red', label='Anchors')
    plt.plot(true_path[:,0], true_path[:,1], 'g-', label='True Path')
    plt.plot(raw[:,0], raw[:,1], 'o', alpha=0.3, label='Distance')
    plt.plot(hybrid[:,0], hybrid[:,1], 'm--', label='Hybrid')
    plt.plot(filtered[:,0], filtered[:,1], 'b-', linewidth=2, label='Filtered')

    plt.legend()
    plt.grid()
    plt.title("NavHive Phase 1")
    plt.show()