# NavHive: Dynamic Anchor-Based Indoor Localization System

## Overview
NavHive is a simulation-based indoor localization system designed to estimate the position of multiple mobile nodes (blind nodes) using signals from multiple anchors. The system demonstrates how dynamic anchor nodes combined with probabilistic filtering and sensor fusion techniques can improve localization accuracy in environments where GPS is unreliable or unavailable.

The project focuses on implementing and visualizing anchor-based positioning using Trilateration, Angle of Arrival (AoA), Adaptive Fusion, and the Extended Kalman Filter (EKF). The system simulates how anchors and target nodes interact in real time, handles anchor health/line-of-sight issues, and visualizes the movement and position estimation through an interactive Streamlit graphical interface.

---

## Features
- **Multi-Node Tracking**: Simulate and track multiple blind nodes simultaneously.
- **Dynamic Anchors**: Anchors can move during simulation, and the system adapts dynamically.
- **Sensor Fusion**: Combines Distance (Trilateration) and Angle of Arrival (AoA) measurements via Adaptive Fusion depending on noise levels.
- **Extended Kalman Filter (EKF)**: Smooths and predicts node trajectories.
- **Anchor Health Monitoring**: Filters out faulty or obstructed anchor measurements to maintain accuracy.
- **Real-Time Dashboard**: Live web-based visualization using Streamlit.
- **Experimentation Suite**: Tools to evaluate Root Mean Square Error (RMSE) against varying sensor noise levels.

---

## System Architecture

### Core Components:
- **`models/`**: Defines the physical entities in the system (`Anchor` and `HiveNode`).
- **`engine/`**: The core mathematical engine containing:
  - Distance and AoA measurement simulation (`measurement.py`, `aoa.py`).
  - Position estimation algorithms (`trilateration.py`, `adaptive_fusion.py`).
  - Health checks for resilient tracking (`anchor_health.py`).
- **`evaluation/`**: Contains metrics computation like RMSE (`metrics.py`).
- **`simulator.py`**: The `NavHiveSimulator` state machine that drives the simulation step-by-step for the dashboard and experiments.

---

## Project Structure

```text
NavHive/
│
├── main.py                 # Static simulation script with Matplotlib visualization
├── dashboard.py            # Streamlit real-time interactive dashboard
├── simulator.py            # Step-by-step simulator engine
├── experiment_runner.py    # Script for evaluating noise vs. localization error
├── hive_manager.py         # Consolidated NavHive execution manager
├── config.py               # Global simulation parameters and constants
│
├── models/
│   ├── anchor.py
│   └── hive_node.py
│
├── engine/
│   ├── adaptive_fusion.py
│   ├── anchor_health.py
│   ├── aoa.py
│   ├── measurement.py
│   └── trilateration.py
│
├── evaluation/
│   └── metrics.py
│
└── README.md
```

---

## Technology Stack

- **Language:** Python
- **Libraries:** NumPy, Matplotlib, Streamlit
- **Techniques:** EKF, Trilateration, AoA, Sensor Fusion

---

## Installation & Usage

### 1. Requirements
Ensure you have Python installed. Install the necessary dependencies:

```bash
pip install numpy matplotlib streamlit scipy
```

### 2. Running the Static Simulation
To run the standard multi-node simulation with a plotted result and RMSE evaluation:

```bash
python main.py
```

### 3. Running the Live Dashboard
To launch the interactive real-time Streamlit dashboard:

```bash
streamlit run dashboard.py
```

### 4. Running Experiments
To evaluate the impact of sensor noise on RMSE:

```bash
python experiment_runner.py
```

---

## Author
Swetabh Suman  
B.Tech Computer Science Engineering

---

## License
This project is developed for educational and research purposes.
