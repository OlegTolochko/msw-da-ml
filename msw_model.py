import numpy as np
from settings import load_settings

type ModelState = np.ndarray[3, int, int]

class ModifiedShallowWaterModel():
    def __init__(self, num_ensemble_members: int):
        '''Implementation of the shallow water model'''
        self.num_ensemble_members = num_ensemble_members

        settings = load_settings()
        config = settings.water_model_config
        self.ngrid = config.num_grid_cells
        self.nsub = config.num_sub_steps
        self.g = config.gravitational_constant
        self.h_cloud = config.h_cloud
        self.h_rain = config.h_rain
        self.phi_cloud = config.phi_cloud
        self.r_gamma = config.r_gamma
        self.time_step_size = config.time_step_size
        self.config = config
        
    def initialize(self, num_init_steps: int = 8):
        '''initializes the initial shallow water model'''
        u, h, r = np.zeros((self.ngrid, self.num_ensemble_members))*3

        u = u + self.config.base_velocity
        h = h + self.config.base_height
        r = r + self.config.base_rain
        init_state = np.concat(u,h,r)

        for step in range(num_init_steps):
            init_state = self.apply_nsub_steps(state=init_state)

        return init_state
        
    def msw_step(self, state_past: np.ndarray, state_present: np.ndarray, phi: np.ndarray, wind_perturbation: np.ndarray):
        '''applies a single state evolution update step
        Args:
            state: leap-frog method prepared water shallow model state with size (3, 3, num_grid_cells, num_ens_members)
            state_past: (3, num_grid_cells, num_ens_members)
            state_present: (3, num_grid_cells, num_ens_members)
            wind_perturbation: noise to be applied to the 
        '''
        u_past, r_past, h_past = state_past
        u_pr, r_pr, h_pr = state_present

        u_pr = u_pr + wind_perturbation

        # trigger convection, if height surpasses height threshold h_cloud
        if h_pr[1:self.ngrid+1] > self.h_cloud:
            phi[0, 1:self.ngrid+1, :] = self.phi_cloud
        else:
            phi[0, 1:self.ngrid+1, :] = self.g*h_pr[1:self.ngrid+1]

        # refill ghost cells
        phi[0, 0, :] = phi[0, self.ngrid, :]
        phi[0, self.ngrid+1, :] = phi[0, 1, :]

        # add rain influence to potential phi
        phi = phi + self.r_gamma*r_pr
        
        # leap frog method derivatives
        u_future = u_past[1:self.ngrid+1] - self.time_step_size/(2*self.ngrid)*(u_pr[2:self.ngrid+2]**2)
        h_future = None
        r_future = None


    def apply_nsub_steps(self, state: np.ndarray):
        '''Applies nsub shallow water model steps to a given state
        Args:
            state: the previous water shallow model state, size (3, num_grid_cells, num_ens_members)
            
        Returns:
            updated_state: Updated state after nsub steps
        '''
        # prepare state for leapfrog method by introducing a past, present and future dimension 
        _, num_grid_cells, num_ens_members = state.shape
        # num_grid_cells+2 to allow for derivatives to be computed for the first and last cell
        new_state = np.zeros((3, num_grid_cells+2, num_ens_members))*3
        new_state[:,*,1:num_grid_cells,:] = state
        new_state[:,*,0,:] = state[:,num_grid_cells-1,:]
        new_state[:,*,num_grid_cells+1,:] = state[:,0,:]

        past_state, present_state, predicted_state = state*3
        phi = np.zeros((1, self.ngrid+2, self.num_ensemble_members))

        for step in range(self.nsub):
            wind_perturbation = self.generate_wind_perturbation(step)
            predicted_state = self.msw_step(present_state, past_state, phi, wind_perturbation)
            past_state = present_state
            present_state = predicted_state
        
        # dimension reduction is still required here
        return new_state
        
    def generate_wind_perturbation(step: int):
        '''generate random wind perturbation'''
        pass

    def save_model_state(save_path='./out/'):
        '''saves model state as .npy (.npz if we ) file'''
        pass

    def load_model_state(load_path='./out/'):
        '''loades model state from .npy file'''
        pass
    
