import numpy as np
from numpy.random import PCG64
from settings import load_settings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ModifiedShallowWaterModel:
    def __init__(
        self, num_ensemble_members: int, random_generator: np.random.Generator
    ):
        """Implementation of the shallow water model"""
        self.num_ensemble_members = num_ensemble_members
        self.random_generator = random_generator

        settings = load_settings()
        config = settings.water_model_config
        self.settings = settings
        self.config = config
        self.gaussian_wind_perturbation = self.generate_gaussian_noise()

        self.current_state = None
        self.full_state_history = []
        self.nsub_state_history = []

    def initialize(self, num_init_steps: int = 8, exclude_from_state_history: bool = True):
        """initializes the initial shallow water model"""
        u = np.zeros((self.config.ngrid, self.num_ensemble_members))
        h = np.zeros((self.config.ngrid, self.num_ensemble_members))
        r = np.zeros((self.config.ngrid, self.num_ensemble_members))

        u = u + self.config.base_velocity
        h = h + self.config.base_height
        r = r + self.config.base_rain
        init_state = np.array([u, h, r])

        for step in range(num_init_steps):
            init_state = self.apply_nsub_steps(state=init_state)

        if exclude_from_state_history:
            self.nsub_state_history = [self.nsub_state_history[-1]]
            self.full_state_history = [self.nsub_state_history[-1]]
        
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

        u_past, h_past, r_past = state_past
        u_pr, h_pr, r_pr = state_present

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

    def apply_nsub_steps(self, state: np.ndarray = None):
        """Applies nsub shallow water model steps to a given state
        Args:
            state: the previous water shallow model state,
                        Shape: (3, num_grid_cells, num_ensemble_members)

        Returns:
            updated_state: Updated state after nsub steps
        """
        if state is None:
            state = self.current_state

        # num_grid_cells+2 to allow for derivatives to be computed for the first and last cell
        past_state = np.zeros((3, self.config.ngrid + 2, self.num_ensemble_members))
        present_state = np.zeros((3, self.config.ngrid + 2, self.num_ensemble_members))
        future_state = np.zeros((3, self.config.ngrid + 2, self.num_ensemble_members))

        # prepare state for leapfrog method by introducing a past, present and future dimension
        past_state[:, 1 : self.config.ngrid + 1] = state
        present_state[:, 1 : self.config.ngrid + 1] = state
        future_state[:, 1 : self.config.ngrid + 1] = state

        phi = np.zeros((self.config.ngrid + 2, self.num_ensemble_members))

        self.full_state_history.append(present_state[:, 1 : self.config.ngrid + 1])
        self.current_state = present_state[:, 1 : self.config.ngrid + 1]

        for step in range(self.config.num_sub_steps):
            wind_perturbation = self.generate_wind_perturbation()
            past_state, present_state, future_state = self.msw_step(
                past_state, present_state, future_state, phi, wind_perturbation
            )
            self.full_state_history.append(present_state[:, 1 : self.config.ngrid + 1])

        self.nsub_state_history.append(
            future_state[:, 1 : self.config.ngrid + 1].copy()
        )
        self.current_state = future_state[:, 1 : self.config.ngrid + 1]
        return future_state[:, 1 : self.config.ngrid + 1]

    def generate_wind_perturbation(self):
        """generate random wind perturbation"""
        wind_perturbation = np.zeros((2 * self.config.ngrid, self.num_ensemble_members))
        gaussian_noise = self.gaussian_wind_perturbation
        for i in range(self.num_ensemble_members):
            pos = self.random_generator.integers(0, self.config.ngrid - 1)
            wind_perturbation[pos : pos + self.config.ngrid, i] = (
                wind_perturbation[pos : pos + self.config.ngrid, i] + gaussian_noise
            )

        return (
            wind_perturbation[0 : self.config.ngrid]
            + wind_perturbation[
                self.config.ngrid : self.config.ngrid + self.config.ngrid
            ]
        )

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

    def get_current_state(self):
        return self.current_state

    def get_full_state_history(self):
        return self.full_state_history

    def get_nsub_state_history(self):
        return self.nsub_state_history

    def update_current_state(self, new_state):
        if (
            self.current_state is not None
            and self.current_state.shape != new_state.shape
        ):
            raise Exception(
                f"Trying to update model with state shapes of {self.current_state.shape}, with a state of shape {new_state.shape}"
            )
        self.current_state = new_state
        self.full_state_history.append(new_state)

    def save_current_model_state(self, save_directory="./out/"):
        """saves model state as .npy (.npz) file"""
        random_state = self.random_generator.bit_generator.state
        nsub_steps = len(self.nsub_state_history)
        base_seed = self.settings.global_config.base_seed
        model_state_name = (
            f"msw_model_ens{self.num_ensemble_members}_{base_seed}_{nsub_steps}.npz"
        )
        full_save_path = f"{save_directory}{model_state_name}"
        np.savez(
            full_save_path,
            current_state=self.current_state,
            nsub_state_history=self.nsub_state_history,
            full_state_history=self.full_state_history,
            num_ensemble_members=self.num_ensemble_members,
            random_state=random_state,
        )

    @classmethod
    def from_state_history(cls, load_path):
        """loads model state from .npz file"""
        with np.load(load_path, allow_pickle=True) as data:
            current_state = data["current_state"]
            nsub_state_history = data["nsub_state_history"]
            full_state_history = data["full_state_history"]
            num_ensemble_members = int(data["num_ensemble_members"])
            random_state = data["random_state"].item()

        bit_gen = PCG64()
        bit_gen.state = random_state
        random_generator = np.random.Generator(bit_gen)

        model = cls(
            num_ensemble_members=num_ensemble_members, random_generator=random_generator
        )
        model.current_state = current_state
        model.nsub_state_history = nsub_state_history
        model.full_state_history = full_state_history

        return model


def animate_evolution_from_history(
    state_history, save_path="./out/model_evolution.mp4"
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
