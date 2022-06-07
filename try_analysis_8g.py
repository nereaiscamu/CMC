# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:03:59 2022

@author: nerea
"""

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
    
    n_disr = np.linspace(8,15,2)
    n_disr_types = ['muted_osc', 'removed_coupling', 'muted_sens', 'all']
    num_reps = 2
    results_cpg_8g1 = np.zeros((len(n_disr)*len(n_disr_types)*num_reps,4))
    results_cpg_8g2 = np.zeros((len(n_disr)*len(n_disr_types)*num_reps,4))
    results_cpg_8g3 = np.zeros((len(n_disr)*len(n_disr_types)*num_reps,4))
    
    for i in range(len(n_disr)):
        for j in range(len(n_disr_types)):
            for k in range (num_reps):
                idx = (i*len(n_disr_types)*num_reps)+(j*num_reps) + k
                data = SalamandraData.from_file('logs/exercise_8g1/simulation_{}.h5'.format(idx))
                with open('logs/exercise_8g1/simulation_{}.pickle'.format(idx), 'rb') as param_file:
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
                links_positions = data.sensors.links.urdf_positions()
                head_positions = links_positions[:, 0, :]
                joints_positions = data.sensors.joints.positions_all()
                joints_velocities = data.sensors.joints.velocities_all()
                #energy_ij= compute_energy(times, joints_positions, joints_velocities)
                distance_ij = compute_distance(head_positions)
                speed_ij = compute_speed(distance_ij, time)
                for index, types in enumerate(n_disr_types):
                    if types == parameters.neural_disr_type:
                        results_cpg_8g1[idx][0] = index
                results_cpg_8g1[idx][1] = len(parameters.disr_segs)
                results_cpg_8g1[idx][2] = distance_ij
                #table[idx][4] = distance_ij/ (np.sum(energy_ij)+0.1)
                results_cpg_8g1[idx][3] = speed_ij
                
    i = -1

    for i in range(len(n_disr)):
        for j in range(len(n_disr_types)):
            for k in range (num_reps):
                idx = (i*len(n_disr_types)*num_reps)+(j*num_reps) + k
                data = SalamandraData.from_file('logs/exercise_8g2/simulation_{}.h5'.format(idx))
                with open('logs/exercise_8g2/simulation_{}.pickle'.format(idx), 'rb') as param_file:
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
                links_positions = data.sensors.links.urdf_positions()
                head_positions = links_positions[:, 0, :]
                joints_positions = data.sensors.joints.positions_all()
                joints_velocities = data.sensors.joints.velocities_all()
                #energy_ij= compute_energy(times, joints_positions, joints_velocities)
                distance_ij = compute_distance(head_positions)
                speed_ij = compute_speed(distance_ij, time)
                for index, types in enumerate(n_disr_types):
                    if types == parameters.neural_disr_type:
                        results_cpg_8g2[idx][0] = index
                results_cpg_8g2[idx][1] = len(parameters.disr_segs)
                results_cpg_8g2[idx][2] = distance_ij
                results_cpg_8g2[idx][3] = speed_ij
                
    i = -1

    for i in range(len(n_disr)):
        for j in range(len(n_disr_types)):
            for k in range (num_reps):
                idx = (i*len(n_disr_types)*num_reps)+(j*num_reps) + k
                data = SalamandraData.from_file('logs/exercise_8g3/simulation_{}.h5'.format(idx))
                with open('logs/exercise_8g3/simulation_{}.pickle'.format(idx), 'rb') as param_file:
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
                links_positions = data.sensors.links.urdf_positions()
                head_positions = links_positions[:, 0, :]
                joints_positions = data.sensors.joints.positions_all()
                joints_velocities = data.sensors.joints.velocities_all()
                #energy_ij= compute_energy(times, joints_positions, joints_velocities)
                distance_ij = compute_distance(head_positions)
                speed_ij = compute_speed(distance_ij, time)
                for index, types in enumerate(n_disr_types):
                    if types == parameters.neural_disr_type:
                        results_cpg_8g3[idx][0] = index
                results_cpg_8g3[idx][1] = len(parameters.disr_segs)
                results_cpg_8g3[idx][2] = distance_ij
                results_cpg_8g3[idx][3] = speed_ij
                
                
    speed8g1_mutosc = results_cpg_8g1[results_cpg_8g1[:,0] == 0][:, [1,3]]
    speed8g1_remcoup = results_cpg_8g1[results_cpg_8g1[:,0] == 1][:, [1,3]]
    speed8g1_mutsens = results_cpg_8g1[results_cpg_8g1[:,0] == 2][:, [1,3]]
    speed8g1_mixed = results_cpg_8g1[results_cpg_8g1[:,0] == 3][:, [1,3]]
    speed8g2_mutosc = results_cpg_8g2[results_cpg_8g2[:,0] == 0][:, [1,3]]
    speed8g2_remcoup = results_cpg_8g2[results_cpg_8g2[:,0] == 1][:, [1,3]]
    speed8g2_mutsens = results_cpg_8g2[results_cpg_8g2[:,0] == 2][:, [1,3]]
    speed8g2_mixed = results_cpg_8g2[results_cpg_8g2[:,0] == 3][:, [1,3]]
    speed8g3_mutosc = results_cpg_8g3[results_cpg_8g3[:,0] == 0][:, [1,3]]
    speed8g3_remcoup = results_cpg_8g3[results_cpg_8g3[:,0] == 1][:, [1,3]]
    speed8g3_mutsens = results_cpg_8g3[results_cpg_8g3[:,0] == 2][:, [1,3]]
    speed8g3_mixed = results_cpg_8g3[results_cpg_8g3[:,0] == 3][:, [1,3]]
    
    results_mutosc = np.zeros((len(n_disr),3))
    for i in range(len(n_disr)):
        g1 = speed8g1_mutosc[speed8g1_mutosc[:,0] == n_disr[i]] [:,1]
        g2 = speed8g2_mutosc[speed8g2_mutosc[:,0] == n_disr[i]] [:,1]
        g3 = speed8g3_mutosc[speed8g3_mutosc[:,0] == n_disr[i]] [:,1]
        results_mutosc[i,0] = np.mean(g1)
        results_mutosc[i,1] = np.mean(g2)
        results_mutosc[i,2] = np.mean(g3)
        
    i = -1   
    results_remcoup = np.zeros((len(n_disr),3))
    for i in range(len(n_disr)):
        g1 = speed8g1_remcoup[speed8g1_remcoup[:,0] == n_disr[i]] [:,1]
        g2 = speed8g2_remcoup[speed8g2_remcoup[:,0] == n_disr[i]] [:,1]
        g3 = speed8g3_remcoup[speed8g3_remcoup[:,0] == n_disr[i]] [:,1]
        results_remcoup[i,0] = np.mean(g1)
        results_remcoup[i,1] = np.mean(g2)
        results_remcoup[i,2] = np.mean(g3)
    i = -1    
    results_mutsens = np.zeros((len(n_disr),3))
    for i in range(len(n_disr)):
        g1 = speed8g1_mutsens[speed8g1_mutsens[:,0] == n_disr[i]] [:,1]
        g2 = speed8g2_mutsens[speed8g2_mutsens[:,0] == n_disr[i]] [:,1]
        g3 = speed8g3_mutsens[speed8g3_mutsens[:,0] == n_disr[i]] [:,1]
        results_mutsens[i,0] = np.mean(g1)
        results_mutsens[i,1] = np.mean(g2)
        results_mutsens[i,2] = np.mean(g3)
    i = -1     
    results_mixed = np.zeros((len(n_disr),3))
    for i in range(len(n_disr)):
        g1 = speed8g1_mixed[speed8g1_mixed[:,0] == n_disr[i]] [:,1]
        g2 = speed8g2_mixed[speed8g2_mixed[:,0] == n_disr[i]] [:,1]
        g3 = speed8g3_mixed[speed8g3_mixed[:,0] == n_disr[i]] [:,1]
        results_mixed[i,0] = np.mean(g1)
        results_mixed[i,1] = np.mean(g2)
        results_mixed[i,2] = np.mean(g3)
    
    
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(n_disr, results_mutosc[:,0])
    ax[0,0].plot(n_disr, results_mutosc[:,1])
    ax[0,0].plot(n_disr, results_mutosc[:,2])
    ax[0,1].plot(n_disr, results_remcoup[:,0])
    ax[0,1].plot(n_disr, results_remcoup[:,1])
    ax[0,1].plot(n_disr, results_remcoup[:,2])
    ax[1,0].plot(n_disr, results_mutsens[:,0])
    ax[1,0].plot(n_disr, results_mutsens[:,1])
    ax[1,0].plot(n_disr, results_mutsens[:,2])
    ax[1,1].plot(n_disr, results_mixed[:,0])
    ax[1,1].plot(n_disr, results_mixed[:,1])
    ax[1,1].plot(n_disr, results_mixed[:,2])
            
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()
        
        
if __name__ == '__main__':
    main(plot=not save_plots())
      

      

