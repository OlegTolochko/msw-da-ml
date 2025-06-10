import numpy as np
from settings import load_settings

class ModifiedShallowWaterModel():
    def __init__(self):
        '''Implementation of the shallow water model'''
        settings = load_settings()
        config = settings.water_model_config
        self.ngrid = config.num_grid_cells
        self.nsub = config.num_sub_steps
        
    def initialize(self, num_ensemble_members: int):
        '''initializes the initial shallow water model'''
        u = np.zeros((self.ngrid,num_ensemble_members))+10.
        h = np.zeros((self.ngrid,num_ensemble_members))+90.
        r = np.zeros((self.ngrid,num_ensemble_members))
        
    def msw_step(self, state: np.ndarray):
        '''applies a single state evolution update step'''
        u_lf, r_lf, h_lf = state
        # msw step here
    
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
        for step in range(self.nsub):
            new_state = self.msw_step(new_state)
        
        # dimension reduction is still required here
        return new_state
        
    def generate_wind_perturbation():
        pass

    def save_model_state(save_path='./out/'):
        '''saves model state as .npy (.npz if we ) file'''
        pass

    def load_model_state(load_path='./out/'):
        '''loades model state from .npy file'''
        pass
    
