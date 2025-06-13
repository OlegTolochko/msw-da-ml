import numpy as np
from settings import load_settings

type ModelState = np.ndarray[3, int, int]


class ModifiedShallowWaterModel:
    def __init__(self, num_ensemble_members: int):
        """Implementation of the shallow water model"""
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
        self.grid_spacing = config.grid_spacing
        self.u_diff_coef = config.u_diff_coef
        self.h_diff_coef = config.h_diff_coef
        self.r_diff_coef = config.r_diff_coef
        self.config = config

    def initialize(self, num_init_steps: int = 8):
        """initializes the initial shallow water model"""
        u, h, r = np.zeros((self.ngrid, self.num_ensemble_members)) * 3

        u = u + self.config.base_velocity
        h = h + self.config.base_height
        r = r + self.config.base_rain
        init_state = np.concat(u, h, r)

        for step in range(num_init_steps):
            init_state = self.apply_nsub_steps(state=init_state)

        return init_state

    def msw_step(
        self,
        state_past: np.ndarray,
        state_present: np.ndarray,
        phi: np.ndarray,
        wind_perturbation: np.ndarray,
    ):
        """
        Applies a single state evolution update step using the leapfrog method.
        Assumption: all input arrays already include ghost cells.

        Args:
            state_past: The model state at time (t - dt),
                        Shape: (3, num_grid_cells + 2, num_ensemble_members)
            state_present: The model state at time (t),
                        Shape: (3, num_grid_cells + 2, num_ensemble_members)
            phi: The potential that largely controls the wind,
                        Shape: (num_grid_cells + 2, num_ensemble_members)
            wind_perturbation: Noise to be applied to the wind field,
                        Shape: (num_grid_cells, num_ensemble_members)

        Returns:
            A numpy array containing the new model state at time (t + dt),
            Shape: (3, num_grid_cells + 2, num_ensemble_members)
        """
        u_past, r_past, h_past = state_past
        u_pr, r_pr, h_pr = state_present

        u_pr[1 : self.ngrid + 1] += wind_perturbation

        # trigger convection, if height surpasses height threshold h_cloud
        phi[1 : self.ngrid + 1] = np.where(
            h_pr[1 : self.ngrid + 1] > self.h_cloud,
            self.phi_cloud,
            self.g * h_pr[1 : self.ngrid + 1],
        )

        # Update ghost cells
        phi[0] = phi[self.ngrid]
        phi[self.ngrid + 1] = phi[1]

        # add rain influence to potential phi
        phi += self.r_gamma * r_pr

        # leap frog method derivatives
        u_advection = -(self.time_step_size / (2 * self.grid_spacing)) * (
            u_pr[2 : self.ngrid + 2] ** 2 - u_pr[0 : self.ngrid] ** 2
        )
        u_pressure_gradient_force = -(2 * self.time_step_size / self.grid_spacing) * (
            phi[1 : self.ngrid + 1] - phi[0 : self.ngrid]
        )

        def calculate_diffusion(past_state_variable, diff_coef):
            return (
                (diff_coef / (4 * (self.grid_spacing**2)))
                * (
                    past_state_variable[2 : self.ngrid + 2]
                    - 2 * past_state_variable[1 : self.ngrid + 1]
                    + past_state_variable[0 : self.ngrid]
                )
                * self.time_step_size
            )

        u_diffusion = calculate_diffusion(u_past, self.u_diff_coef)
        u_future = (
            u_past[1 : self.ngrid + 1]
            + u_advection
            + u_pressure_gradient_force
            + u_diffusion
        )

        h_mass_divergence = (self.time_step_size / self.grid_spacing) * (
            u_pr[2 : self.ngrid + 2]
            * (h_pr[1 : self.ngrid + 1] + h_pr[2 : self.ngrid + 2])
            - u_pr[1 : self.ngrid + 1]
            * (h_pr[0 : self.ngrid] + h_pr[1 : self.ngrid + 1])
        )
        h_diffusion = calculate_diffusion(h_past, self.h_diff_coef)

        h_future = h_mass_divergence + h_diffusion

        r_diffusion = calculate_diffusion(r_past, self.r_diff_coef)

        r_future = None

    def apply_nsub_steps(self, state: np.ndarray):
        """Applies nsub shallow water model steps to a given state
        Args:
            state: the previous water shallow model state,
                        Shape: (3, num_grid_cells, num_ensemble_members)

        Returns:
            updated_state: Updated state after nsub steps
        """
        # num_grid_cells+2 to allow for derivatives to be computed for the first and last cell
        past_state, present_state, predicted_state = (
            np.zeros((3, self.ngrid + 2, self.num_ensemble_members)) * 3
        )

        # prepare state for leapfrog method by introducing a past, present and future dimension
        (
            past_state[:, 1 : self.ngrid + 1],
            present_state[:, 1 : self.ngrid + 1],
            predicted_state[:, 1 : self.ngrid + 1],
        ) = state * 3

        phi = np.zeros((self.ngrid + 2, self.num_ensemble_members))

        predicted_state = None
        for step in range(self.nsub):
            wind_perturbation = self.generate_wind_perturbation(step)
            predicted_state = self.msw_step(
                present_state, past_state, phi, wind_perturbation
            )
            past_state = present_state
            present_state = predicted_state

        return predicted_state[:, 1 : self.ngrid + 1]

    def generate_wind_perturbation(step: int):
        """generate random wind perturbation"""
        pass

    def save_model_state(save_path="./out/"):
        """saves model state as .npy (.npz if we ) file"""
        pass

    def load_model_state(load_path="./out/"):
        """loades model state from .npy file"""
        pass
