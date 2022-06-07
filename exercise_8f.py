"""Exercise 8f"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8f(timestep):
    """Exercise 8f"""
    parameter_set = [
        SimulationParameters(
            duration=20,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive_mlr= 3.5,  # An example of parameter part of the grid search
            amplitudes= 1.3,  # Showed to be efficient 
            #phase_lag = 1 ,  # Works well for forward swimming and with 
            #the initial phases so that it moves without coupling
            sensory_feedback = True,
            decoupled = False,
            sensory_feedback_gain = sens_gains* np.ones(16),
            coupling_gain = coupling_gains,

        )
        for sens_gains in np.linspace(-100, -0.1, 20)
        for coupling_gains in np.linspace(0.1, 10, 20)

    ]

    # Grid search
    os.makedirs('./logs/exercise_8f/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8f/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=True,  # For fast mode (not real-time)
            #initial_phases = np.concatenate([np.linspace(2*np.pi, 0, 8), np.pi + np.linspace(2*np.pi, 0, 8), np.zeros(4)]),
            
            headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)


if __name__ == '__main__':
    exercise_8f(timestep=1e-2)