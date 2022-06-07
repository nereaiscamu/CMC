"""Plot results"""

import pickle
import numpy as np
from pkg_resources import resource_listdir
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
import os

from scipy.integrate import ode, odeint # I put it here

def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)


def plot_1d(ax, subplot, results, labels, title):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 2].

    labels - The labels should be a list of two string for the xlabel and the
    ylabel (in that order).
    """

    ax[subplot].plot(results[0,:], results[1,:], marker='.')

    ax[subplot].set_xlabel(labels[0])
    ax[subplot].set_ylabel(labels[1])
    ax[subplot].set_title(title)
    ax[subplot].set_ylim([-0.1, 0.4])
    ax[subplot].grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='none',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(plot=True):
    """Main"""
    # Load data
    i = -1
    results = []


    for files in os.listdir('logs/8g_CPG_no_coupling'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_CPG_no_coupling/simulation_{}.h5'.format(i))
        with open('logs/8g_CPG_no_coupling/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    fig, ax = plt.subplots(4,3, constrained_layout=False)
    fig.tight_layout()
    plot_1d(ax,(1,0),results,["Number of disruptions","Speed"],'CPG with muted couplings')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_CPG_no_oscillators'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_CPG_no_oscillators/simulation_{}.h5'.format(i))
        with open('logs/8g_CPG_no_oscillators/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(2,0),results,["Number of disruptions","Speed"],'CPG with muted oscillators')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_CPG_no_sensors'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_CPG_no_sensors/simulation_{}.h5'.format(i))
        with open('logs/8g_CPG_no_sensors/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(0,0),results,["Number of disruptions","Speed"],'CPG with muted sensors')


    i = -1
    results = []


    for files in os.listdir('logs/8g_CPG_mixed'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_CPG_mixed/simulation_{}.h5'.format(i))
        with open('logs/8g_CPG_mixed/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(3,0),results,["Number of disruptions","Speed"],'CPG with mixed disruptions')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_sensory_no_coupling'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_sensory_no_coupling/simulation_{}.h5'.format(i))
        with open('logs/8g_sensory_no_coupling/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(1,1),results,["Number of disruptions","Speed"],'Decoupled with muted couplings')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_sensory_no_oscillators'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_sensory_no_oscillators/simulation_{}.h5'.format(i))
        with open('logs/8g_sensory_no_oscillators/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(2,1),results,["Number of disruptions","Speed"],'Decoupled with muted oscillators')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_sensory_no_sensors'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_sensory_no_sensors/simulation_{}.h5'.format(i))
        with open('logs/8g_sensory_no_sensors/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(0,1),results,["Number of disruptions","Speed"],'Decoupled with muted sensors')


    i = -1
    results = []


    for files in os.listdir('logs/8g_sensory_mixed'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_sensory_mixed/simulation_{}.h5'.format(i))
        with open('logs/8g_sensory_mixed/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(3,1),results,["Number of disruptions","Speed"],'Decoupled with mixed disruptions')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_combined_no_coupling'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_combined_no_coupling/simulation_{}.h5'.format(i))
        with open('logs/8g_combined_no_coupling/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(1,2),results,["Number of disruptions","Speed"],'Combined with muted couplings')
    
    
    i = -1
    results = []


    for files in os.listdir('logs/8g_combined_no_oscillators'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_combined_no_oscillators/simulation_{}.h5'.format(i))
        with open('logs/8g_combined_no_oscillators/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(2,2),results,["Number of disruptions","Speed"],'Combined with muted oscillators')

    
    i = -1
    results = []


    for files in os.listdir('logs/8g_combined_no_sensors'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_combined_no_sensors/simulation_{}.h5'.format(i))
        with open('logs/8g_combined_no_sensors/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(0,2),results,["Number of disruptions","Speed"],'Combined with muted sensors')


    i = -1
    results = []


    for files in os.listdir('logs/8g_combined_mixed'):
        if files.endswith('.h5'):
            i = i+1
        data = SalamandraData.from_file('logs/8g_combined_mixed/simulation_{}.h5'.format(i))
        with open('logs/8g_combined_mixed/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        timestep = data.timestep
        n_iterations = np.shape(data.sensors.links.array)[0]
        times = np.arange(
            start=0,
            stop=timestep*n_iterations,
            step=timestep,
        )
        timestep = times[1] - times[0]
        links_positions = data.sensors.links.urdf_positions()
        head_positions = links_positions[:, 0, :]
        speed = (np.linalg.norm(head_positions[-1])-np.linalg.norm(head_positions[0]))/10
        if files.endswith('.h5'):
            results.append(speed)
            print("Speed = "+str(speed))

    results = np.reshape(results, (8,10))
    results = np.mean(results, axis=1).T
    results = np.vstack([np.arange(0,8,1),results])
    plot_1d(ax,(3,2),results,["Number of disruptions","Speed"],'Combined with mixed disruptions')
    

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=True)

