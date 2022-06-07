"""Simulation parameters"""
import numpy as np

class SimulationParameters:
    """Simulation parameters"""

    def __init__(self, **kwargs):
        super(SimulationParameters, self).__init__()
        # Default parameters
        self.n_body_joints = 8
        self.n_legs_joints = 4
        self.duration = 30
        self.phase_lag = None
        #self.amplitude_gradient = None
        self.drive = 4
        self.amplitude_gradient = None
        
        self.set_seed = False
        self.randseed = 0
        self.n_disruption_couplings = 0
        self.n_disruption_oscillators = 0
        self.n_disruption_sensors = 0
        
        self.sensory_feedback = False        
        self.sensory_feedback_gain = np.zeros(16)
        self.decoupling = False
        

        # Feel free to add more parameters (ex: MLR drive)
        # self.drive_mlr = ...
        # ...
        # Update object with provided keyword arguments
        # NOTE: This overrides the previous declarations
        self.__dict__.update(kwargs)