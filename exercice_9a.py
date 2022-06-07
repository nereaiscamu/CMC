"""Exercise 9a1"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_9a1(timestep):
    """Exercise 9a1"""

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=15,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=2,
        )
        # for drive in np.linspace(3, 4, 2)
        # for phase_lag in np.linspace(0,np.pi,2)
        # for amplitudes in ...
        # for ...
    ]

    # Grid search
    os.makedirs('./logs/example/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/example/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try! #or "water"
            fast=False,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)



if __name__ == '__main__':
    exercise_9a1(timestep=1e-2)

