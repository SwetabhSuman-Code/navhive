# NavHive: Hybrid Indoor Localization Simulation

## Overview
NavHive is a simulation framework demonstrating advanced indoor localization techniques for environments where GPS is unavailable. The system estimates the position of multiple blind nodes moving within an operating area, utilizing fixed anchor stations.

This project specifically highlights how **Hybrid Probabilistic Algorithms** can combine noisy distance (e.g., RSSI/ToF) and angular (AoA) sensor measurements in real-time. By utilizing **Adaptive Sensor Fusion** and an **Extended Kalman Filter (EKF)**, the system smooths out measurement spikes to track moving nodes with high accuracy.

---

## 🔬 Core Algorithms

The NavHive system processes sensor data through a four-stage pipeline during each time step:

### 1. Distance-Based Trilateration (Least Squares)
Given the distance between a blind node and multiple anchors, the system creates intersecting geometric circles. To handle inevitable sensor noise, it uses a **Linear Least Squares optimization** strategy to algebraically find the coordinate with the minimum squared error across all overlapping circles.

### 2. Angle of Arrival (AoA)
The system calculates the theoretical angle (phase difference) from each anchor to the node. After injecting simulated angular hardware error, the system uses **Trigonometric Ray Intersection** ($y = mx + c$) to mathematically solve for where the noisy rays cross.

### 3. Adaptive Sensor Fusion
NavHive calculates a dynamic weight for both the Trilateration and AoA estimates based on inversely proportional noise levels ($W = 1 / Noise$). The final, high-fidelity `hybrid_est` is produced by blending the two coordinates based on which sensor type is currently providing cleaner data.

### 4. Continuous State Tracking (Extended Kalman Filter)
Finally, the `hybrid_est` is passed into a 4-dimensional **Extended Kalman Filter (EKF)** tracker running a State Vector of $(x, y, v_x, v_y)$. By calculating the Kalman Gain, it constantly balances predicted physics movement (momentum) against the raw noisy sensor updates to eliminate jitter and plot a smooth, continuous trajectory.

---

## 🛠️ Project Structure

```text
NavHive/
│
├── main.py                 # Static simulation script with Matplotlib visualization
├── dashboard.py            # Streamlit real-time interactive dashboard
├── simulator.py            # Step-by-step simulator engine
├── experiment_runner.py    # Script for evaluating noise vs. localization error
├── hive_manager.py         # Consolidated NavHive execution manager
├── config.py               # Global simulation parameters (Currently set to 10 Iterations)
│
├── models/
│   ├── anchor.py           # Fixed reference points
│   └── hive_node.py        # Blind nodes with internal EKF trackers
│
├── engine/
│   ├── adaptive_fusion.py  # Blending function
│   ├── anchor_health.py    # Line-of-sight obstruction checker
│   ├── aoa.py              # Angle of Arrival logic
│   ├── measurement.py      # RSSI/Distance noise injection
│   └── trilateration.py    # Least Squares intersection logic
│
├── evaluation/
│   └── metrics.py          # RMSE calculations
│
└── README.md
```

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. Install the necessary mathematical and visualization dependencies:

```bash
pip install numpy matplotlib streamlit scipy
```

### 1. Running the Standard Simulation
To run the standard multi-node simulation (configured for 10 iterations) and view the plotted tracked paths and RMSE errors:

```bash
python main.py
```

### 2. View the Live Interactive Dashboard
To launch the real-time simulation visualization in your browser:

```bash
streamlit run dashboard.py
```

### 3. Run Performance Evaluation
To examine how increasing sensor noise (RSSI interference) directly spikes the overall Root Mean Square Error (RMSE):

```bash
python experiment_runner.py
```

---

## 👨‍💻 Author
**Swetabh Suman**  
*B.Tech Computer Science Engineering*

---
*Developed for academic research and educational demonstration in autonomous tracking methodologies.*
