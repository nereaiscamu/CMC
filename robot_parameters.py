"""Robot parameters"""

import numpy as np
from farms_core import pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([self.n_oscillators,self.n_oscillators])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.sensory_feedback = False
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i
        self.set_feedback_gain(parameters)
        #self.set_oscillation_amplitudes(parameters)
        


    def set_feedback_gain(self, parameters):
        gain = np.zeros(16)
        if hasattr(parameters,"sensory_feedback"):
            self.sensory_feedback = parameters.sensory_feedback
            if parameters.sensory_feedback == True:
                gain[:] = parameters.sensory_feedback_gain[:]
                if hasattr(parameters,"neural_disr") and parameters.neural_disr == True:
                    if parameters.neural_disr_type == 'muted_sens':
                        for i in range(len(parameters.disr_segs)):
                            idx = int(parameters.disr_segs[i])
                            gain[idx] = 0
                    if parameters.neural_disr_type == 'all':
                        for i in range(len(parameters.disr_segs)):
                            idx = int(parameters.disr_segs[i])
                            if i in [2,5,8,11,14]:
                                gain[idx] = 0
            self.sensory_feedback_gain = gain   
            

    def limb_freq(self,drive):
        if drive < 1:
            return 0
        elif drive < 3:
            return 0.2*drive
        else:
            return 0

    def limb_amp(self,drive):
        if drive < 1:
            return 0
        elif drive < 3:
            return 0.131*drive + 0.131
        else:
            return 0

    def body_freq(self,drive):
        if drive < 1:
            return 0
        elif drive < 5:
            return 0.2*drive + 0.3
        else:
            return 0

    def body_amp(self,drive):
        if drive < 1:
            return 0
        elif drive < 5:
            return 0.065*drive + 0.196
        else:
            return 0
        

    def set_frequencies(self, parameters):
        """Set frequencies"""
        self.freqs = 3*np.ones(20)
        if hasattr(parameters,"drive"):   
            if(isinstance(parameters.drive,int) or isinstance(parameters.drive,float)):
                for i in range(16):
                    self.freqs[i] = self.body_freq(parameters.drive)
                for i in range(16,20):
                    self.freqs[i] = self.limb_freq(parameters.drive)
            elif(isinstance(parameters.drive,list)):
                for i in range(8):
                    self.freqs[i] = self.body_freq(parameters.drive[0])
                    self.freqs[i+8] = self.body_freq(parameters.drive[1])
                for i in range(16,18):
                    self.freqs[i] = self.limb_freq(parameters.drive[0])
                    self.freqs[i+2] = self.limb_freq(parameters.drive[1]) 
         
        if hasattr(parameters,"neural_disr") and parameters.neural_disr == True:
            if parameters.neural_disr_type == 'muted_osc':
                for i in range(len(parameters.disr_segs)):
                    idx = int(parameters.disr_segs[i])
                    self.freqs[idx] = 0
            if parameters.neural_disr_type == 'all':
                for i in range(len(parameters.disr_segs)):
                    idx = int(parameters.disr_segs[i])
                    if i in [0,3,6,9,12, 15]:
                        self.freqs[idx] = 0
                        
            #if hasattr(parameters,"freqs"):
                #self.freqs = np.ones(self.n_oscillators) * parameters.freqs
                
                
                


    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        self.coupling_weights = np.zeros((20, 20))
        if hasattr(parameters, "decoupled") and parameters.decoupled==True:
            return
        # else, ...
        # body joints to body joints:
        for i in range(8):
            self.coupling_weights[i, i + 8] = 10  # i+8 on i
            self.coupling_weights[i + 8, i] = 10  # i on i+8
            if i > 0:
                self.coupling_weights[i - 1, i] = 10  # i on i-1
                self.coupling_weights[i + 8 - 1, i + 8] = 10
            if i < 7:
                self.coupling_weights[i + 1, i] = 10  # i on i+1
                self.coupling_weights[i + 8 + 1, i + 8] = 10

        # limb joints to limb + body joints

        # limb n°1: index 16
        self.coupling_weights[17, 16] = 10
        self.coupling_weights[18, 16] = 10
        for i in range(4):
            self.coupling_weights[i, 16] = 30
        # limb n°2: index 17
        self.coupling_weights[16, 17] = 10
        self.coupling_weights[19, 17] = 10
        for i in range(8, 12):
            self.coupling_weights[i, 17] = 30
        # limb n°3: index 18
        self.coupling_weights[16, 18] = 10
        self.coupling_weights[19, 18] = 10
        for i in range(4, 8):
            self.coupling_weights[i, 18] = 30
        # limb n°4: index 19
        self.coupling_weights[17, 19] = 10
        self.coupling_weights[18, 19] = 10
        for i in range(12, 16):
            self.coupling_weights[i, 19] = 30

                
                
        if hasattr(parameters,"drive"):
            print(type(parameters.drive))
            if(isinstance(parameters.drive,int) or isinstance(parameters.drive,float)):
                if parameters.drive >= 3:
                    self.coupling_weights[:,16:20] = 0
            elif(isinstance(parameters.drive,list)):
                if parameters.drive[0] >= 3:
                    self.coupling_weights[:,[16,18]] = 0
                if parameters.drive[1] >= 3:
                    self.coupling_weights[:,[17,19]] = 0
                
        if hasattr(parameters, "coupling_gain"):
            self.coupling_weights[:,16:20] = self.coupling_weights[:,16:20]
            self.coupling_weights[:, 0:16] = self.coupling_weights[:, 0:16]*parameters.coupling_gain
        
        if hasattr(parameters,"neural_disr") and parameters.neural_disr == True:
            if parameters.neural_disr_type == 'removed_coupling':
                for i in range(len(parameters.disr_segs)):
                    idx = int(parameters.disr_segs[i])
                    self.coupling_weights[:, idx] = 0
                    self.coupling_weights[idx, :] = 0
            if parameters.neural_disr_type == 'all':
                for i in range(len(parameters.disr_segs)):
                    idx = int(parameters.disr_segs[i])
                    if i in [1,4,7,10,13]:
                        self.coupling_weights[:, idx] = 0
                        self.coupling_weights[idx, :] = 0




    def set_phase_bias(self, parameters):
        """Set phase bias"""
        self.phase_bias = np.zeros((20, 20))
        if hasattr(parameters,"phase_lag") and parameters.phase_lag is not None:
            base_bias = parameters.phase_lag
        else:
            base_bias = np.pi/4
        self.phase_bias = np.zeros((20, 20))
        # body joints to body joints:
        for i in range(8):
            self.phase_bias[i, i + 8] = np.pi  # like in the paper
            self.phase_bias[i + 8, i] = np.pi  # like in the paper
            if i > 0:
                self.phase_bias[i - 1, i] = - base_bias
                self.phase_bias[i + 8 - 1, i + 8] = - base_bias
            if i < 7:
                self.phase_bias[i + 1, i] = base_bias
                self.phase_bias[i + 8 + 1, i + 8] = base_bias

        # limb joints to limb + body joints

        # limb n°1: index 16
        self.phase_bias[17, 16] = np.pi
        self.phase_bias[18, 16] = np.pi
        self.phase_bias[0:4, 16] = np.pi
        # limb n°2: index 17
        self.phase_bias[16, 17] = np.pi
        self.phase_bias[19, 17] = np.pi
        self.phase_bias[8:12, 17] = np.pi
        # limb n°3: index 8
        self.phase_bias[16, 18] = np.pi
        self.phase_bias[19, 18] = np.pi
        self.phase_bias[4:8, 18] = np.pi
        # limb n°4: index 19
        self.phase_bias[17, 19] = np.pi
        self.phase_bias[18, 19] = np.pi
        self.phase_bias[12:16, 19] = np.pi
        #pylog.warning('Phase bias must be set')
        

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates = 20 * np.ones(20)  # as in the paper

        #pylog.warning('Convergence rates must be set')

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        self.nominal_amplitudes = 0.05*np.ones(20) # Value not found
        if hasattr(parameters,"drive"):
            if(isinstance(parameters.drive,int) or isinstance(parameters.drive,float)):
                # limb frequencies
                for i in range(16):
                    self.nominal_amplitudes[i] = self.body_amp(parameters.drive)
                for i in range(16, 20):
                    self.nominal_amplitudes[i] = self.limb_amp(parameters.drive)
            elif(isinstance(parameters.drive,list)):
                for i in range(8):
                    self.nominal_amplitudes[i] = self.body_amp(parameters.drive[0])
                    self.nominal_amplitudes[i+8] = self.body_amp(parameters.drive[1])
                for i in range(16, 18):
                    self.nominal_amplitudes[i] = self.limb_amp(parameters.drive[0])
                    self.nominal_amplitudes[i+2] = self.limb_amp(parameters.drive[1])

        if hasattr(parameters,"amplitudes"): 
                self.nominal_amplitudes= parameters.amplitudes * self.nominal_amplitudes 
                    
        if hasattr(parameters, 'amp_gradient'):
            self.nominal_amplitudes[:8] = self.nominal_amplitudes[:8] * np.linspace(parameters.amp_gradient[0], parameters.amp_gradient[1], 8)
            self.nominal_amplitudes[8:16] = self.nominal_amplitudes[8:16]* np.linspace(parameters.amp_gradient[0], parameters.amp_gradient[1], 8)

                 
