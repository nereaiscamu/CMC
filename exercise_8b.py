"""Exercise 8b"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8b(timestep):
    """Exercise 8b"""
 # Parameters
    parameter_set = [
        SimulationParameters(
            duration=15,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            amplitudes = amplitude,  # Just an example
            phase_lag=phase_lag,  # or np.zeros(n_joints) for example
            turn=0,  # Another example
            drive=3,
        )
        for amplitude in np.linspace(0.5, 2, 20)
        for phase_lag in np.linspace(-1.5, 0.5, 30)
    ]

    # Grid search
    os.makedirs('./logs/exercise_8b/n_20x30_zoom', exist_ok=True)
    idx = 0
    for simulation_i, sim_parameters in enumerate(parameter_set):
        if idx >=0:
            filename = './logs/exercise_8b/n_20x30_zoom/simulation_{}.{}'
            print(simulation_i)
            sim, data = simulation(
                sim_parameters=sim_parameters,  # Simulation parameters, see above
                arena='water',  # Can also be 'ground', give it a try!
                fast=True,  # For fast mode (not real-time)
                headless=True,  # For headless mode (No GUI, could be faster)
            )
            # Log robot data
            data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
            # Log simulation parameters
            with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
                pickle.dump(sim_parameters, param_file)
        idx += 1
    # Use exercise_example.py for reference
    pass


if __name__ == '__main__':
    exercise_8b(timestep=1e-2)

