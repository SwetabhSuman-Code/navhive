import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

from simulator import NavHiveSimulator


st.title("NavHive Live Localization Dashboard")

num_nodes = st.slider("Number of Nodes", 1, 10, 5)

sim = NavHiveSimulator(num_nodes=num_nodes)

chart = st.empty()

fig, ax = plt.subplots()

for step in range(100):

    positions = sim.step(step)

    ax.clear()

    # current anchor positions
    anchors = np.array([a.position for a in sim.anchors])

    ax.scatter(
        anchors[:,0],
        anchors[:,1],
        c="red",
        s=120,
        label="Anchors"
    )

    # plot node positions
    for p in positions:

        true_pos = p["true"]
        pred_pos = p["filtered"]

        ax.scatter(true_pos[0], true_pos[1], c="green")
        ax.scatter(pred_pos[0], pred_pos[1], c="blue")

    ax.set_title("NavHive Real-Time Tracking")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.legend()

    chart.pyplot(fig)

    time.sleep(0.1)