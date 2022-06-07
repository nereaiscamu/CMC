# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:40:02 2022

@author: nerea
"""


""" 8f result analysis"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
from scipy import integrate
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import math

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')



from plot_results import plot_trajectory
from plot_results import plot_positions
from plot_results import plot_2d
from plot_results import compute_energy
from plot_results import compute_distance
from plot_results import plot_3D


def compute_speed(distance, time):
    speed = distance/ time 
    return speed


def main(plot=True):
    """Main"""
    # Load data
    
    sens_gains = np.linspace(-100, -0.1, 20)
    coupling_gains = np.linspace(0.1, 10, 20)
    table = np.zeros((len(sens_gains)*len(coupling_gains),6))
    
    for i in range(len(sens_gains)):
        for j in range(len(coupling_gains)):
            idx = i*len(coupling_gains)+j
            data = SalamandraData.from_file('logs/exercise_8f/simulation_{}.h5'.format(idx))
            with open('logs/exercise_8f/simulation_{}.pickle'.format(idx), 'rb') as param_file:
                parameters = pickle.load(param_file)
            timestep = data.timestep
            n_iterations = np.shape(data.sensors.links.array)[0]
            times = np.arange(
                start=0,
                stop=timestep*n_iterations,
                step=timestep,
            )
            time = parameters.duration
            timestep = times[1] - times[0]
            sens_gain = parameters.sensory_feedback_gain 
            coupling_gain = parameters.coupling_gain
            links_positions = data.sensors.links.urdf_positions()
            head_positions = links_positions[:, 0, :]
            joints_positions = data.sensors.joints.positions_all()
            joints_velocities = data.sensors.joints.velocities_all()
            energy_ij= compute_energy(times, joints_positions, joints_velocities)
            distance_ij = compute_distance(head_positions)
            speed_ij = compute_speed(distance_ij, time)
            table[idx][0] = np.unique(sens_gain)
            table[idx][1] = 10 * coupling_gain
            table[idx][2] = np.sum(energy_ij)
            table[idx][3] = distance_ij
            table[idx][4] = distance_ij/ (np.sum(energy_ij)+0.1)
            table[idx][5] = speed_ij
            
            
    plt.figure('Energy')
    plot_2d(table[:,[0,1,2]], labels=['Sensory gains','Coupling weights','Energy'], n_data=300, log=False, cmap=None)
    plt.figure('Speed')
    plot_2d(table[:,[0,1,5]], labels=['Sensory gains','Coupling weights', 'Speed(m/s)'], n_data=300, log=False, cmap=None)

    # Show plots
    if plot:
        plt.show()
        
    else:
        save_figures()
        
        
if __name__ == '__main__':
    main(plot=not save_plots())