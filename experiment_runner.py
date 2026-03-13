import numpy as np
import matplotlib.pyplot as plt
from simulator import NavHiveSimulator

def run_noise_experiment():

    noise_levels = [0.1,0.3,0.5,0.7,1.0]
    rmse_results = []

    for noise in noise_levels:

        sim = NavHiveSimulator(num_nodes=5)
        errors = []

        for step in range(60):

            positions = sim.step(step)

            for p in positions:

                true_pos = p["true"]
                pred_pos = p["filtered"]

                error = np.linalg.norm(true_pos - pred_pos)
                errors.append(error)

        rmse = np.sqrt(np.mean(np.array(errors)**2))
        rmse_results.append(rmse)

    plt.plot(noise_levels, rmse_results)
    plt.xlabel("Sensor Noise")
    plt.ylabel("RMSE Error")
    plt.title("Noise vs Localization Error")
    plt.show()

run_noise_experiment()