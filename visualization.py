import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from settings import load_settings

settings = load_settings()
global_config = settings.global_config

animation_path = f"{global_config.out_path}{global_config.animation_out_filename}"
os.makedirs(animation_path, exist_ok=True)


class ModelComparatorVisualizer:
    def __init__(
        self, history1: list, history2: list, model1_name="CNN", model2_name="Truth"
    ):
        self.hist1 = history1
        self.hist2 = history2
        self.name1 = model1_name
        self.name2 = model2_name

        if len(self.hist1) != len(self.hist2):
            raise ValueError("Histories must have the same length.")

        self.num_frames = len(self.hist1)
        self.ngrid = self.hist1[0].shape[1]
        self.x_axis = np.arange(self.ngrid)

        self.time_steps = []
        self.rmse_history = []
        self.mass_error_hist1 = []
        self.mass_error_hist2 = []
        self.initial_mass1 = np.sum(self.hist1[0][1].mean(axis=1))
        self.initial_mass2 = np.sum(self.hist2[0][1].mean(axis=1))

        self.fig, self.axs = plt.subplots(
            2, 2, figsize=(16, 9), constrained_layout=True
        )
        self.fig.suptitle("Model Comparison Dashboard", fontsize=16)

        # 1. state evolution
        self.ax1 = self.axs[0, 0]
        (self.line1,) = self.ax1.plot([], [], lw=2, label=self.name1)
        (self.line2,) = self.ax1.plot([], [], lw=1.5, ls="--", label=self.name2)
        self.fill_spread = None
        self.ax1.set_title("State Evolution (Water Height h)")
        self.ax1.set_xlabel("Grid Cell")
        self.ax1.set_ylabel("Height (h)")
        self.ax1.grid(True)
        self.ax1.legend()

        # 2. water height difference
        self.ax2 = self.axs[0, 1]
        (self.line_error,) = self.ax2.plot([], [], lw=2, color="red", label="Error")
        self.ax2.axhline(0, color="black", lw=0.5)
        self.ax2.set_title("Error (h)")
        self.ax2.set_xlabel("Grid Cell")
        self.ax2.set_ylabel("Difference")
        self.ax2.grid(True)

        # 3. RSME
        self.ax3 = self.axs[1, 0]
        (self.line_rmse,) = self.ax3.plot([], [], "o-", label="RMSE(h)")
        self.ax3.set_title("Root Mean Square Error over Time")
        self.ax3.set_xlabel("Time Step")
        self.ax3.set_ylabel("RMSE")
        self.ax3.grid(True)

        # 4. mass conservation error
        self.ax4 = self.axs[1, 1]
        (self.line_mass1,) = self.ax4.plot(
            [], [], "o-", label=f"{self.name1} Mass Error"
        )
        (self.line_mass2,) = self.ax4.plot(
            [], [], "o--", label=f"{self.name2} Mass Error"
        )
        self.ax4.set_title("Mass Conservation Error (%)")
        self.ax4.set_xlabel("Time Step")
        self.ax4.set_ylabel("Error (%)")
        self.ax4.grid(True)
        self.ax4.legend()

        self.fig.subplots_adjust(bottom=0.1)

        self._set_initial_limits()

    def _set_initial_limits(self):
        all_h1 = [s[1].mean(axis=1) for s in self.hist1]
        all_h2 = [s[1].mean(axis=1) for s in self.hist2]
        h_min = min(np.min(all_h1), np.min(all_h2)) * 0.995
        h_max = max(np.max(all_h1), np.max(all_h2)) * 1.005
        self.ax1.set_ylim(h_min, h_max)

        all_errors = [
            s1[1].mean(axis=1) - s2[1].mean(axis=1)
            for s1, s2 in zip(self.hist1, self.hist2)
        ]
        err_max = np.max(np.abs(all_errors)) * 1.1
        self.ax2.set_ylim(-err_max, err_max)
        self.ax2.set_xlim(0, self.ngrid)

    def _update(self, frame):
        state1 = self.hist1[frame]
        state2 = self.hist2[frame]

        h1_mean = state1[1].mean(axis=1)
        h2_mean = state2[1].mean(axis=1)
        h1_std = state1[1].std(axis=1)

        # 1: Add lines to state evolution
        self.line1.set_data(self.x_axis, h1_mean)
        self.line2.set_data(self.x_axis, h2_mean)
        if self.fill_spread:
            self.fill_spread.remove()
        self.fill_spread = self.ax1.fill_between(
            self.x_axis,
            h1_mean - h1_std,
            h1_mean + h1_std,
            color="skyblue",
            alpha=0.4,
            label="Spread",
        )

        # 2: compute and update error
        error = h1_mean - h2_mean
        self.line_error.set_data(self.x_axis, error)

        # update time series metrics
        self.time_steps.append(frame)
        rmse = np.sqrt(np.mean(error**2))
        self.rmse_history.append(rmse)

        mass1 = np.sum(h1_mean)
        mass2 = np.sum(h2_mean)
        self.mass_error_hist1.append(
            100 * (mass1 - self.initial_mass1) / self.initial_mass1
        )
        self.mass_error_hist2.append(
            100 * (mass2 - self.initial_mass2) / self.initial_mass2
        )

        # 3: update RSME
        self.line_rmse.set_data(self.time_steps, self.rmse_history)
        self.ax3.relim()
        self.ax3.autoscale_view()

        # 4: Update mass error
        self.line_mass1.set_data(self.time_steps, self.mass_error_hist1)
        self.line_mass2.set_data(self.time_steps, self.mass_error_hist2)
        self.ax4.relim()
        self.ax4.autoscale_view()

        return (
            self.line1,
            self.line2,
            self.fill_spread,
            self.line_error,
            self.line_rmse,
            self.line_mass1,
            self.line_mass2,
        )

    def animate(self, save_name: str):
        save_name = save_name.removesuffix(".pth")

        save_path = f"{animation_path}{save_name}_{self.name1}_vs_{self.name2}_{len(self.hist1)}.mp4"
        anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self.num_frames,
            blit=True,
            interval=100,
        )
        anim.save(save_path, writer="ffmpeg", fps=10, dpi=150)
        plt.close(self.fig)
        print(f"Dashboard animation saved to {save_path}")
