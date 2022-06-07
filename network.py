"""Oscillator network ODE"""

import numpy as np
from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters, loads=None):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters

    Returns
    -------
    :<np.array>
        Returns derivative of state (phases and amplitudes)

    """
    n_oscillators = robot_parameters.n_oscillators
    Freqs = robot_parameters.freqs
    Weights = robot_parameters.coupling_weights
    Biases = robot_parameters.phase_bias
    Nom_amplitudes = robot_parameters.nominal_amplitudes
    Rates = robot_parameters.rates


    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    # Implement equation here

    state_deriv = np.concatenate([np.zeros_like(phases), np.zeros_like(amplitudes)])
    for i in range(n_oscillators):
        theta_i_deriv = 2*np.pi*Freqs[i]
        for j in range(n_oscillators):
            theta_i_deriv += amplitudes[j]*Weights[i,j]*np.sin(phases[j]-phases[i]-Biases[i,j])
        r_i_deriv = Rates[i]*(Nom_amplitudes[i]-amplitudes[i])
        state_deriv[i] = theta_i_deriv
        state_deriv[n_oscillators+i] = r_i_deriv

    if robot_parameters.sensory_feedback == True:

        sensory_feedback_gain = robot_parameters.sensory_feedback_gain
        state_deriv[:8] += sensory_feedback_gain[:8] * loads[:8] * np.cos(phases[:8])
        state_deriv[8:16] += sensory_feedback_gain[8:16] *-1*loads[:8] * np.cos(phases[8:16])

    return state_deriv


def motor_output(phases, amplitudes, iteration):
    """Motor output

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    : <np.array>
        Motor outputs for joint in the system.

    """
    # Implement equation here    
    mot_out = np.zeros_like(phases)[:8]
    mot_out_l = np.zeros(4)
    mot_out = amplitudes[:8]*(1+np.cos(phases[:8]))-amplitudes[8:16]*(1+np.cos(phases[8:16]))

    for i in range(4):
        if np.sin(phases[16+i]) >= 0:
            k = round(phases[i+16]//(2*np.pi))
            mot_out_l[i] = 11*phases[i+16]/10 - (np.pi+k*2*np.pi)/10 
        #if np.abs(np.cos(phases[16+i])) > np.abs(np.cos(4*np.pi/5)) and np.cos(phases[16+i]) < np.cos(2*np.pi):
        else:
            m = round(phases[i+16]//(2*np.pi))
            k = m+1
            mot_out_l[i] = 4*phases[i+16]/5 + (2*k*np.pi)/5 - np.pi/10

    return np.concatenate([mot_out,mot_out_l])

class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations, state, initial_phases = None):
        super().__init__()
        self.n_iterations = n_iterations
        # States
        self.state = state
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        # Replace your oscillator phases here
        if initial_phases is not None:
            self.state.set_phases(
                iteration=0,
                value=initial_phases)
        else:
            self.state.set_phases(
                iteration=0,
                value=2*np.random.ranf(self.robot_parameters.n_oscillators),
            )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state.array[0], t=0.0)

    def step(self, iteration, time, timestep, loads=None):
        """Step"""
        if iteration + 1 >= self.n_iterations:
            return
        self.solver.set_f_params(self.robot_parameters, loads)
        self.state.array[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here
        if (iteration != None):
            outputs = np.zeros(self.robot_parameters.n_oscillators)
            outputs = self.state.amplitudes(iteration = iteration) * (1 + np.cos(self.state.phases(iteration = iteration)))
        else:
            outputs = np.zeros_like(self.state.amplitudes())
            outputs = self.state.amplitudes()* (1 + np.cos(self.state.phases()))
        return outputs

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        return motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )
        
    def get_output(self, iteration=None):
        """Get motor position"""
        return self.outputs(
            iteration=iteration,
        )

