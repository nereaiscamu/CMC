"""Plot results"""

import pickle
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from salamandra_simulation.data import SalamandraData
from salamandra_simulation.parse_args import save_plots
from salamandra_simulation.save_figures import save_figures
#import seaborn as sns
import math

def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)

def compute_travelling_distance(positions):
    """Plot positions"""
    """
    iterations = positions.shape[0]
    positions = np.array(positions)
    iter_distances = np.sum((positions[1:,:] - positions[:-1,:])**2, axis=1)
    # distance starting from the middle of the simulation
    distance = np.sum(iter_distances[iterations//2:])
    """
    positions = np.array(positions)
    distance = np.sum((positions[-1,:] - positions[0,:])**2)
    return distance

def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)

def compute_energy(timestep, joints_velocities, joints_torques):
    min_energy = 0*np.ones_like(joints_velocities)
    powers = np.array(joints_velocities)*np.array(joints_torques)
    powers = np.maximum(min_energy,powers)
    n_steps = powers.shape[0] # number of iterations
    energy = np.sum(powers[n_steps//2:,:])*timestep
    # sum on every joint (8) and at each time
    return energy


def compute_distance(positions):
    start = np.zeros(3)
    end = np.zeros(3)
    for i, point in enumerate(positions.T):
        start[i] = point[0]
        end[i] = point[-1]
    return math.dist(start,end)

def threshold_energy(energy, mini, maxi):
    return np.max([np.min([energy, maxi]), mini])

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
        method='nearest'  # nearest, cubic
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

def plot_3D(results, heat = False):
    x = results[:,0]
    y = results[:,1]
    z = results[:,2]
    fig = plt.figure()
    X, Y = np.meshgrid(x,y)
    print("X",X)
    print("Y",Y)
    if heat == True:
        nx = X.max() - X.min() + 1
        ny = Y.max() - Y.min() + 1
        Z = np.zeros((nx,ny)) 
        #assert X.shape == Y.shape == Z.shape
        for i in range(len(X)):
            Z[X[i]-X.min()][Y[i]-Y.min()] = z[i]
        Nspacingx = 0.2
        Nspacingy = 0.2
        figure_name = 'Heatmap_Energy'
        #plt.pcolor(np.arange(nx),np.arange(ny),Z,cmap=plt.cm.Reds)
        plt.pcolor(np.arange(nx), np.arange(ny), Z, cmap=plt.cm.Reds)
        plt.colorbar()
        plt.xlim(0,X.max()-X.min())
        plt.ylim(0,Y.max()- Y.min())
        xlabels = np.arange(X.min(),X.max(),Nspacingx) # define Nspacing accordingly 
        ylabels = np.arange(Y.min(),Y.max(),Nspacingy) 
        plt.xticks(np.arange(0,X.max()-X.min(),Nspacingx),xlabels)
        plt.yticks(np.arange(0,Y.max()-Y.min(),Nspacingy),ylabels)
        plt.savefig(figure_name,dpi=400)
        
        
    else :
        figure_name = '3Dplot_Energy'
        Z = results[:,2]
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.savefig(figure_name,dpi=400)
    plt.show()

def main(plot=True):
    """Main"""

    files_path = "logs/exercise_8b/n_20x30_zoom/"
    # refer to what was initiated in exercice_8b
    namps = 20
    nphases = 30
    Energy_map = np.zeros((namps*nphases,3))
    distance_map = np.zeros((namps*nphases,3))
    Efficacity_map = np.zeros((namps*nphases,3))
    idx = 0
    for i in range(namps):
        for j in range(nphases):
                data = SalamandraData.from_file(files_path + f'simulation_{idx}.h5')
                with open(files_path + f'simulation_{idx}.pickle', 'rb') as param_file:
                    parameters = pickle.load(param_file)
                timestep = data.timestep
                n_iterations = np.shape(data.sensors.links.array)[0]
                times = np.arange(
                    start=0,
                    stop=timestep*n_iterations,
                    step=timestep,
                )
                amplitudes = parameters.amplitudes
                phase_lag = parameters.phase_lag
                osc_phases = data.state.phases()
                osc_amplitudes = data.state.amplitudes()
                links_positions = data.sensors.links.urdf_positions()
                head_positions = links_positions[:, 0, :]
                tail_positions = links_positions[:, 8, :]
                joints_positions = data.sensors.joints.positions_all()
                joints_velocities = data.sensors.joints.velocities_all()
                joints_torques = data.sensors.joints.motor_torques_all()
                energy_ij = compute_energy(timestep, joints_velocities, joints_torques)
                distance_ij = compute_travelling_distance(head_positions)
                Energy_map[idx,0] = amplitudes
                Energy_map[idx,1] = phase_lag
                Energy_map[idx,2] = energy_ij

                distance_map[idx,0] = amplitudes
                distance_map[idx,1] = phase_lag
                distance_map[idx,2] = distance_ij


                Efficacity_map[idx, 0] = amplitudes
                Efficacity_map[idx, 1] = phase_lag
                efficacity = distance_ij/energy_ij
                Efficacity_map[idx, 2] = threshold_energy(efficacity, mini=1e-4, maxi = 5e-3)

                idx += 1


    # Notes:
    # For the links arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]

    # Plot data
    plt.figure('Positions')
    plot_positions(times, head_positions)
    plt.figure('Trajectory')
    plot_trajectory(head_positions)

    plt.figure('Efficacity')
    plot_2d(Efficacity_map, labels=['amplitudes', 'phase lags', 'Efficacity'], n_data=300, log=False, cmap=None)
    plt.figure('Energy')
    plot_2d(Energy_map, labels=['amplitudes','phase lags','Energy'], n_data=300, log=False, cmap=None)
    plt.figure('Distance')
    plot_2d(distance_map, labels=['amplitudes','phase lags','distance travelled'], n_data=300, log=False, cmap=None)

    #plt.figure('Energy 3D')
    #plot_3D(Energy_map, heat=True)
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

