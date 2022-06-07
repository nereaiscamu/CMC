"""Exercise 8g"""

import os
import pickle
import numpy as np
from salamandra_simulation.simulation import simulation
from simulation_parameters import SimulationParameters

 

neural_dis_seg_matrix = np.zeros((10,16))

for i in range(10):
    random_seg_list = np.array(np.random.permutation(16))
    neural_dis_seg_matrix[i,:] = random_seg_list
    
""" HERE I TRY TO BUILD A MATRIX TO HAVE A BIT MORE CONTROL ABOUT HOW THE 
NEURAL DISRUPTIONS ARE APPLIED TO EACH MODEL

    
I CREATE HERE THE LIST FOR THE NEURAL DISRUPTIONS TYPES, THEN FOR EACH 
SIMULATION ONLY A SINGLE VALUE IS PASSED
STILL NEED TO IMPLEMENT HOW TO APPLY DIFFERENT DISRUPTIONS 
FOR EACH SIMULATION WE WOULD HAVE A NUMBER OF DISRUPTED SEGMENTS, 
AND EVERYTIME A NEW SEGMENT IS DISRUPTED IT SHOULD BE WITH A DIFFERENT TYPE
MAYBE WE CAN DO LIKE IF NUM = 1,4,7,10,13 --> USE MUTED OSC...
CAREFUL, DEPENDING ON THE MODEL SOME DISRUPTIONS MAKE NO SENSE """

   
neural_disr_types = ['muted_osc', 'removed_coupling', 'muted_sens', 'all']
    

def exercise_8g1(timestep):
    """Exercise 8g1"""
       # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive= 3.5,  # An example of parameter part of the grid search
            sensory_feedback = False,
            decoupled = False,
            neural_disr = True,
            neural_disr_type = types,
            disr_segs = neural_dis_seg_matrix[j,:int(num)],

            #sensory_feedback_gain = 0.5,
            # ...
        )
        for types in neural_disr_types
        for num in np.linspace(0,15,8)
        for j in range(10)
    ]

    # Grid search
    os.makedirs('./logs/exercise_8g1/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8g1/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='water',  # Can also be 'ground', give it a try!
            fast=True,  # For fast mode (not real-time)
            headless=True,  # For headless mode (No GUI, could be faster)
            #initial_phases = np.concatenate([np.linspace(2*np.pi, 0, 8), np.pi + np.linspace(2*np.pi, 0, 8), np.zeros(4)]),
            # record=True,  # Record video
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
    # Use exercise_example.py for reference

def exercise_8g2(timestep):
    """Exercise 8g2"""

       # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive= 3.5,  # An example of parameter part of the grid search
            sensory_feedback = True,
            decoupled = True,
            sensory_feedback_gain = -20*np.ones(16),
            neural_disr = True,
            neural_disr_type = types,
            disr_segs = neural_dis_seg_matrix[1,:int(num)],

            # ...
        )
        for types in neural_disr_types
        for num in np.linspace(0,15,8)
        for j in range(10)
    ]

    # Grid search
    os.makedirs('./logs/exercise_8g2/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8g2/simulation_{}.{}'
        sim, data= simulation(
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
    # Use exercise_example.py for reference


def exercise_8g3(timestep):
    """Exercise 8g3"""
       # Parameters
    parameter_set = [
        SimulationParameters(
            duration=10,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, 0],  # Orientation in Euler angles [rad]
            drive= 3.5,  # An example of parameter part of the grid search
            sensory_feedback = True,
            decoupled = False,
            sensory_feedback_gain = -20*np.ones(16),
            neural_disr = True,
            neural_disr_type = types,# could be 1,2,3,all
            disr_segs = neural_dis_seg_matrix[1,:int(num)],

        )
        for types in neural_disr_types
        for num in np.linspace(0,15,8)
        for j in range(10)
    ]

    # Grid search
    os.makedirs('./logs/exercise_8g3/', exist_ok=True)
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/exercise_8g3/simulation_{}.{}'
        sim, data  = simulation(
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
    # Use exercise_example.py for reference



if __name__ == '__main__':
    #exercise_8g1(timestep=1e-2)
    exercise_8g2(timestep=1e-2)
    exercise_8g3(timestep=1e-2)



