"""Exercise 8c"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8c(timestep):
    """Exercise 8c"""
    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive= 3.5,  # An example of parameter part of the grid search
            amp_gradient = [Rhead, Rtail],
            freqs = 1,  # we are asked to fix 1Hz as frequency
            # ...
        )
        for Rhead in np.linspace(0.1, 3, 15)
        for Rtail in np.linspace(0.1, 3, 15)
        # for ...
    ]

    # Grid search
    os.makedirs('./logs/exercise_8c/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8c/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    # Use exercise_example.py for reference


if __name__ == '__main__':
    exercise_8c(timestep=1e-2)


