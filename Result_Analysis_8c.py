# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:42:18 2022

@author: nerea
"""

""" 8c result analysis"""

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
    
    # Rhead = np.linspace(0.5,2, 15)
    # Rtail = np.linspace(0.5 ,2, 15)
    Rhead = np.linspace(0.1, 3, 15)
    Rtail = np.linspace(0.1, 3, 15)
    table = np.zeros((len(Rhead)*len(Rtail),6))
    
    for i in range(len(Rhead)):
        for j in range(len(Rtail)):
            idx = i*len(Rtail)+j
            data = SalamandraData.from_file('logs/exercise_8c/simulation_{}.h5'.format(idx))
            with open('logs/exercise_8c/simulation_{}.pickle'.format(idx), 'rb') as param_file:
                parameters = pickle.load(param_file)
            timestep = data.timestep
            n_iterations = np.shape(data.sensors.links.array)[0]
            times = np.arange(
                start=0,
                stop=timestep*n_iterations,
                step=timestep,
            )
            drive = parameters.drive
            time = parameters.duration
            timestep = times[1] - times[0]
            R_head = parameters.amp_gradient[0] 
            R_tail = parameters.amp_gradient[1]
           # osc_phases = data.state.phases()
            #osc_amplitudes = data.state.amplitudes()
            links_positions = data.sensors.links.urdf_positions()
            head_positions = links_positions[:, 0, :]
            #tail_positions = links_positions[:, 8, :]
            joints_positions = data.sensors.joints.positions_all()
            joints_velocities = data.sensors.joints.velocities_all()
            #joints_torques = data.sensors.joints.motor_torques_all()
            energy_ij= compute_energy(times, joints_positions, joints_velocities)
            distance_ij = compute_distance(head_positions)
            speed_ij = compute_speed(distance_ij, time)
            table[idx][0] = R_head #* (0.196 + (0.065*drive))
            table[idx][1] = R_tail #* (0.196 + (0.065*drive))
            table[idx][2] = np.sum(energy_ij)
            table[idx][3] = distance_ij
            table[idx][4] = distance_ij/ (np.sum(energy_ij)+1)
            table[idx][5] = speed_ij
            
            
            
    plt.figure('Energy')
    plot_2d(table[:,[0,1,2]], labels=['Rhead (x nominal amplitude)','Rtail (x nominal amplitude)','Energy'], n_data=300, log=False, cmap=None)
    plt.figure('Distance')
    plot_2d(table[:,[0,1,3]], labels=['Rhead (x nominal amplitude)','Rtail (x nominal amplitude)','Distance (m)'], n_data=300, log=False, cmap=None)
    plt.figure('Efficiency')
    plot_2d(table[:,[0,1,4]], labels=['Rhead (x nominal amplitude)','Rtail (x nominal amplitude)','Efficiency'], n_data=300, log=False, cmap=None)
    plt.figure('Speed')
    plot_2d(table[:,[0,1,5]], labels=['Rhead (x nominal amplitude)','Rtail (x nominal amplitude)','Speed(m/s)'], n_data=300, log=False, cmap=None)

    # Show plots
    if plot:
        plt.show()
        
    else:
        save_figures()
        
        
if __name__ == '__main__':
    main(plot=not save_plots())