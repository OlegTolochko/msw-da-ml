import numpy as np
from settings import load_settings


class ModifiedShallowWaterModel:
    def __init__(
        self, num_ensemble_members: int, random_generator: np.random.Generator
    ):
        """Implementation of the shallow water model"""
        self.num_ensemble_members = num_ensemble_members
        self.random_generator = random_generator

        settings = load_settings()
        config = settings.water_model_config
        self.config = config
        self.gaussian_wind_perturbation = self.generate_gaussian_noise()

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
        state_future: np.ndarray,
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
            state_future: The model state to be predicted at time (t + dt),
                        Shape: (3, num_grid_cells + 2, num_ensemble_members)
            phi: The potential that largely controls the wind,
                        Shape: (num_grid_cells + 2, num_ensemble_members)
            wind_perturbation: Noise to be applied to the wind field,
                        Shape: (num_grid_cells, num_ensemble_members)

        Returns:
            A numpy array containing the new model state at time (t + dt),
            Shape: (3, num_grid_cells + 2, num_ensemble_members)
        """

        # Update ghost cells
        state_past[:, 0] = state_past[:, self.config.ngrid]
        state_present[:, 0] = state_present[:, self.config.ngrid]
        state_future[:, 0] = state_future[:, self.config.ngrid]
        state_past[:, self.config.ngrid + 1] = state_past[:, 1]
        state_present[:, self.config.ngrid + 1] = state_present[:, 1]
        state_future[:, self.config.ngrid + 1] = state_future[:, 1]

        u_past, r_past, h_past = state_past
        u_pr, r_pr, h_pr = state_present

        u_pr[1 : self.config.ngrid + 1] += wind_perturbation

        # trigger convection, if height surpasses height threshold h_cloud
        phi[1 : self.config.ngrid + 1] = np.where(
            h_pr[1 : self.config.ngrid + 1] > self.config.h_cloud,
            self.config.phi_cloud,
            self.config.gravitational_constant * h_pr[1 : self.config.ngrid + 1],
        )

        # Update phi ghost cells
        phi[0] = phi[self.config.ngrid]
        phi[self.config.ngrid + 1] = phi[1]

        # add rain influence to potential phi
        phi += self.config.r_gamma * r_pr

        def calculate_diffusion(past_state_variable, diff_coef):
            return (
                (diff_coef / (4 * (self.config.grid_spacing**2)))
                * (
                    past_state_variable[2 : self.config.ngrid + 2]
                    - 2 * past_state_variable[1 : self.config.ngrid + 1]
                    + past_state_variable[0 : self.config.ngrid]
                )
                * self.config.time_step_size
            )

        # leap frog method derivatives
        # Update velocity/wind u
        u_advection = -(self.config.time_step_size / (2 * self.config.grid_spacing)) * (
            u_pr[2 : self.config.ngrid + 2] ** 2 - u_pr[0 : self.config.ngrid] ** 2
        )
        u_pressure_gradient_force = -(
            2 * self.config.time_step_size / self.config.grid_spacing
        ) * (phi[1 : self.config.ngrid + 1] - phi[0 : self.config.ngrid])

        u_diffusion = calculate_diffusion(u_past, self.config.u_diff_coef)
        u_future = (
            u_past[1 : self.config.ngrid + 1]
            + u_advection
            + u_pressure_gradient_force
            + u_diffusion
        )

        # Update height/mass h
        h_mass_divergence = (self.config.time_step_size / self.config.grid_spacing) * (
            u_pr[2 : self.config.ngrid + 2]
            * (h_pr[1 : self.config.ngrid + 1] + h_pr[2 : self.config.ngrid + 2])
            - u_pr[1 : self.config.ngrid + 1]
            * (h_pr[0 : self.config.ngrid] + h_pr[1 : self.config.ngrid + 1])
        )
        h_diffusion = calculate_diffusion(h_past, self.config.h_diff_coef)
        h_future = h_past[1 : self.ngrid + 1] + h_mass_divergence + h_diffusion

        # Update rain r
        rain_production_mask = np.logical_and(
            h_pr[1 : self.config.ngrid + 1] > self.config.h_rain,
            u_pr[2 : self.config.ngrid] - u_pr[1 : self.config.ngrid + 1] < 0,
        )
        rain_production_rate = np.where(rain_production_mask, self.config.r_rate, 0)
        r_removal = (
            -self.config.r_removal_rate
            * self.time_step_size
            * 2
            * r_pr[1 : self.ngrid + 1]
        )
        r_production = (
            -2
            * rain_production_rate
            * (self.time_step_size / self.grid_spacing)
            * u_pr[2 : self.config.ngrid + 2]
            - u_pr[1 : self.config.ngrid + 1]
        )
        r_diffusion = calculate_diffusion(r_past, self.config.r_diff_coef)
        r_future = r_past[1 : self.ngrid + 1] + r_removal + r_production + r_diffusion

        state_future[0, 1 : self.config.ngrid + 1] = u_future
        state_future[1, 1 : self.config.ngrid + 1] = h_future
        state_future[2, 1 : self.config.ngrid + 1] = r_future

        # rain is not allowed to be 0
        state_future[2] = np.where(state_future[2] < 0.0, 0.0, state_future[2])

        # Solution stabalization
        second_derivative = (
            self.config.filter_coeff
            * 0.5
            * (state_future - 2 * state_present + state_past)
        )

        next_state_past = (
            state_present + self.config.filter_correction_past * second_derivative
        )
        next_state_present = (
            state_future + self.config.filter_correction_present * second_derivative
        )

        return next_state_past, next_state_present, state_future

    def apply_nsub_steps(self, state: np.ndarray):
        """Applies nsub shallow water model steps to a given state
        Args:
            state: the previous water shallow model state,
                        Shape: (3, num_grid_cells, num_ensemble_members)

        Returns:
            updated_state: Updated state after nsub steps
        """
        # num_grid_cells+2 to allow for derivatives to be computed for the first and last cell
        past_state, present_state, future_state = (
            np.zeros((3, self.config.ngrid + 2, self.num_ensemble_members)) * 3
        )

        # prepare state for leapfrog method by introducing a past, present and future dimension
        (
            past_state[:, 1 : self.config.ngrid + 1],
            present_state[:, 1 : self.config.ngrid + 1],
            future_state[:, 1 : self.config.ngrid + 1],
        ) = state * 3

        phi = np.zeros((self.config.ngrid + 2, self.num_ensemble_members))

        for step in range(self.config.num_sub_steps):
            wind_perturbation = self.generate_wind_perturbation(step)
            past_state, present_state, future_state = self.msw_step(
                past_state, present_state, future_state, phi, wind_perturbation
            )

        return future_state[:, 1 : self.config.ngrid + 1]

    def generate_wind_perturbation(self, step: int):
        """generate random wind perturbation"""
        wind_perturbation = np.zeros(2 * self.config.ngrid, self.num_ensemble_members)
        gaussian_noise = self.gaussian_wind_perturbation
        for i in range(self.num_ensemble_members):
            pos = self.random_generator.randint(0, self.config.ngrid - 1)
            wind_perturbation[pos : pos + self.config.ngrid, i] = (
                wind_perturbation[pos : pos + self.config.ngrid, i] + gaussian_noise
            )

        return wind_perturbation

    def generate_gaussian_noise(self):
        noise_center = float((self.config.ngrid + 1) / 2)
        x_axis = np.array(range(self.config.ngrid + 1))
        std = float(
            self.config.wind_perturbation_standard_deviation
        )  # Standard deviation for gaussian, width of perturbation
        amp = float(
            self.config.wind_perturbation_noise_amplitude
        )  # Amplitude of the added noise field (in m/s)
        gaussian = (1 / (std * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((x_axis - noise_center) / std) ** 2
        )
        perturbation = (
            gaussian[1 : self.config.ngrid + 1] - gaussian[0 : self.config.ngrid]
        )  # derivative of gaussian
        perturbation_normalized = amp * perturbation / max(perturbation)

        return perturbation_normalized

    def save_model_state(save_path="./out/"):
        """saves model state as .npy (.npz) file"""
        pass

    def load_model_state(load_path="./out/"):
        """loades model state from .npy file"""
        pass
