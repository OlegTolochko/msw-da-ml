"""Physical shallow water model"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from msw_da_ml.settings import load_settings


class EnsembleModel:
    def __init__(
        self, num_ensemble_members: int, random_generator: np.random.Generator
    ):
        """Implementation of the shallow water model"""
        self.num_ensemble_members = num_ensemble_members
        self.random_generator = random_generator
        settings = load_settings()
        self.config = settings.water_model_config

        self.physics_engine = ShallowWaterPhysics(self.config, num_ensemble_members)
        self.gaussian_wind_perturbation = self._generate_gaussian_noise()

        self.state = None
        self.history = []

    def initialize(self, num_init_steps: int = 8):
        """initializes the initial ensemble state"""
        u = np.zeros((self.config.ngrid, self.num_ensemble_members))
        h = np.zeros((self.config.ngrid, self.num_ensemble_members))
        r = np.zeros((self.config.ngrid, self.num_ensemble_members))

        u = u + self.config.base_velocity
        h = h + self.config.base_height
        r = r + self.config.base_rain
        self.state = np.array([u, h, r])

        wind_perturbations = self._generate_wind_perturbation_sequence()
        for step in range(num_init_steps):
            self.propagate(wind_perturbations=wind_perturbations)

        self.history.clear()
        self.history.append(self.state.copy())
        return self.state

    def assimilate(self, new_state: np.ndarray):
        """Updates the models current state with an assimilated state."""
        if self.state.shape != new_state.shape:
            raise ValueError("Shape of new state does not match current state.")

        self.state = new_state
        if self.history:
            self.history[-1] = self.state.copy()
        else:
            raise ValueError("The current class instance has no history to correct.")

    def propagate(self, wind_perturbations: np.ndarray | None = None):
        """Propagates the model state forward by nsub_steps"""
        if wind_perturbations is None:
            wind_perturbations = self._generate_wind_perturbation_sequence()

        past = self.physics_engine.add_ghost_cells(self.state)
        present = self.physics_engine.add_ghost_cells(self.state)
        for wind_perturbation in wind_perturbations:
            past, present, future = self.physics_engine.step(
                past, present, wind_perturbation, has_ghost_cells=True
            )

        self.state = future[:, 1:-1]
        self.history.append(self.state.copy())
        return self.state

    def _generate_wind_perturbation_sequence(self):
        return np.stack(
            [
                self._generate_wind_perturbation()
                for _ in range(self.config.num_sub_steps)
            ],
            axis=0,
        )

    def _generate_wind_perturbation(self):
        """generate random wind perturbation"""
        wind_perturbation = np.zeros((2 * self.config.ngrid, self.num_ensemble_members))
        gaussian_noise = self.gaussian_wind_perturbation
        for i in range(self.num_ensemble_members):
            pos = self.random_generator.integers(0, self.config.ngrid)
            wind_perturbation[pos : pos + self.config.ngrid, i] = (
                wind_perturbation[pos : pos + self.config.ngrid, i] + gaussian_noise
            )

        return (
            wind_perturbation[0 : self.config.ngrid]
            + wind_perturbation[
                self.config.ngrid : self.config.ngrid + self.config.ngrid
            ]
        )

    def _generate_gaussian_noise(self):
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

    def get_state(self):
        return self.state

    def get_history(self):
        return self.history


class ShallowWaterPhysics:
    def __init__(self, config, num_ensemble_members: int):
        self.config = config
        self.num_ensemble_members = num_ensemble_members

    def step(
        self,
        state_past: np.ndarray,
        state_present: np.ndarray,
        wind_perturbation: np.ndarray,
        has_ghost_cells: bool = False,
    ):
        """
        Applies a single state evolution update step using the leapfrog method.

        Args:
            state_past: The model state at time (t - dt),
                        Shape: (3, num_grid_cells + 2, num_ensemble_members)
            state_present: The model state at time (t),
                        Shape: (3, num_grid_cells + 2, num_ensemble_members)
            wind_perturbation: Noise to be applied to the wind field,
                        Shape: (num_grid_cells, num_ensemble_members)

        Returns:
            past, present and future states,
            Shape: (3, num_grid_cells, num_ensemble_members)
        """
        if not has_ghost_cells:
            state_past = self.add_ghost_cells(state_past)
            state_present = self.add_ghost_cells(state_present)

        # Update ghost cells
        state_future = np.zeros_like(state_present)

        u_past, h_past, r_past = state_past
        u_pr, h_pr, r_pr = state_present

        u_pr[1 : self.config.ngrid + 1] += wind_perturbation

        # Potential that largely controls the wind
        phi = np.zeros((self.config.ngrid + 2, self.num_ensemble_members))

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
        h_future = h_past[1 : self.config.ngrid + 1] - h_mass_divergence + h_diffusion

        # Update rain r
        rain_production_mask = np.logical_and(
            h_pr[1 : self.config.ngrid + 1] > self.config.h_rain,
            u_pr[2 : self.config.ngrid + 2] - u_pr[1 : self.config.ngrid + 1] < 0,
        )
        rain_production_rate = np.where(rain_production_mask, self.config.r_rate, 0)
        r_removal = (
            -self.config.r_removal_rate
            * self.config.time_step_size
            * 2
            * r_pr[1 : self.config.ngrid + 1]
        )
        r_production = (
            -2
            * rain_production_rate
            * (self.config.time_step_size / self.config.grid_spacing)
            * (u_pr[2 : self.config.ngrid + 2] - u_pr[1 : self.config.ngrid + 1])
        )
        r_diffusion = calculate_diffusion(r_past, self.config.r_diff_coef)
        r_future = (
            r_past[1 : self.config.ngrid + 1] + r_removal + r_production + r_diffusion
        )

        state_future[0, 1 : self.config.ngrid + 1] = u_future
        state_future[1, 1 : self.config.ngrid + 1] = h_future
        state_future[2, 1 : self.config.ngrid + 1] = r_future
        state_future[:, 0] = state_future[:, self.config.ngrid]
        state_future[:, self.config.ngrid + 1] = state_future[:, 1]

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

        if has_ghost_cells:
            return next_state_past, next_state_present, state_future

        return next_state_past[:, 1:-1], next_state_present[:, 1:-1], state_future[:, 1:-1]

    def add_ghost_cells(self, state):
        state_ghost = np.zeros((3, self.config.ngrid + 2, self.num_ensemble_members))

        state_ghost[:, 1 : self.config.ngrid + 1] = state
        state_ghost[:, 0] = state[:, self.config.ngrid - 1]
        state_ghost[:, -1] = state[:, 0]
        return state_ghost


def animate_evolution_from_history(
    state_history: list, save_path: str = "./out/model_evolution.mp4"
):
    ngrid = state_history[0].shape[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(ngrid)

    (line,) = ax.plot([], [], lw=2, label="Water Height (h)")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    stats_text = ax.text(0.02, 0.05, "", transform=ax.transAxes, fontsize=12)

    all_h_values = [s[1].mean(axis=1) for s in state_history]
    h_min = np.min(all_h_values) * 0.99
    h_max = np.max(all_h_values) * 1.01
    ax.set_ylim(h_min, h_max)
    ax.set_xlim(0, ngrid - 1)
    ax.set_title("Shallow Water Model State Evolution")
    ax.set_xlabel("Grid Cell")
    ax.set_ylabel("Water Height (h)")
    ax.legend()
    ax.grid(True)

    def update(frame):
        full_state = state_history[frame]

        state_data = full_state.mean(axis=2)
        u, h, r = state_data
        line.set_data(x_axis, h)
        time_text.set_text(f"Time Step: {frame}/{len(state_history)}")

        stats_str = (
            f"Mean Velocity (u): {u.mean():.3f} m/s\nMean Rain (r):     {r.mean():.4f}"
        )
        stats_text.set_text(stats_str)

        return line, time_text, stats_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(state_history),
        blit=True,
        interval=50,
    )

    anim.save(save_path, writer="ffmpeg", fps=5)
