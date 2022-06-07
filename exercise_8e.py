"""Exercise 8e"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters


def exercise_8e1(timestep):
    """Exercise 8e1"""

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=3.5,
            decoupled=True,
            amplitudes = 1.5,
            sensory_feedback=False,
        )

    ]

    # Grid search
    os.makedirs('./logs/8e1/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8e1/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=False,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            #initial_phases = np.concatenate([np.linspace(2*np.pi,0,8), np.pi + np.linspace(2*np.pi,0,8),np.zeros(4)]),
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)


def exercise_8e1_2(timestep):
    """Exercise 8e1_2"""

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=3.5,
            decoupled=True,
            amplitudes = 1.5,
            sensory_feedback=False,
        )

    ]

    # Grid search
    os.makedirs('./logs/8e1_2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8e1_2/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=False,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            initial_phases = np.concatenate([np.linspace(2*np.pi,0,8), np.pi + np.linspace(2*np.pi,0,8),np.zeros(4)]),
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)




def exercise_8e2(timestep):
    """Exercise 8e2"""

    # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive=3.5,
            amplitudes = 1.3,
            decoupled=True,
            sensory_feedback=True,
            sensory_feedback_gain = -20* np.ones(16),
        )
    ]

    os.makedirs('./logs/8e2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8e2/simulation_{}.{}'
        initial_phases = np.concatenate([np.linspace(2*np.pi,0,8), np.pi + np.linspace(2*np.pi,0,8),np.zeros(4)])

        print("wanted initial phases: ",initial_phases)
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=False,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            #initial_phases = initial_phases,
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)


if __name__ == '__main__':
    #exercise_8e1(timestep=1e-2)
    #exercise_8e1_2(timestep=1e-2)
    exercise_8e2(timestep=1e-2)

