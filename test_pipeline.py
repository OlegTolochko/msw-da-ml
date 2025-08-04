import os
import copy
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt

from inference import load_normalization, load_trained_model
from msw_model import EnsembleModel
from assimilation import EnsembleKalmanFilter, QPEnsemble
from observation_generation import ObservationGenerator
from random_manager import RandomGenerators
from settings import load_settings

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

experiments_path = (
    f"{global_config.out_path}{global_config.experiment_histories_out_filename}"
)
os.makedirs(experiments_path, exist_ok=True)


@dataclass
class ExperimentHistory:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis: List[np.ndarray]
    enkf_background: List[np.ndarray]
    qpens_background: List[np.ndarray]
    cnn_background: List[np.ndarray]
    seed: int


class ExperimentPipeline:
    def __init__(self, load_model_name: str = ""):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.loaded_model_name = load_trained_model(
            load_model_name, self.device
        )
        self.norm_stats = load_normalization(self.loaded_model_name, self.device)

    def run_single_experiment(
        self, seed: int, num_inference_steps: int
    ) -> ExperimentHistory:
        rngs = RandomGenerators.from_seed(seed)

        truth_model = EnsembleModel(
            num_ensemble_members=1, random_generator=rngs.truth_rng
        )
        truth_model.initialize()

        enkf_model = EnsembleModel(
            num_ensemble_members=experiment_config.num_ensemble_members,
            random_generator=rngs.ensemble_rng,
        )
        enkf_model.initialize()

        qpens_model = copy.deepcopy(enkf_model)
        cnn_model = copy.deepcopy(enkf_model)

        enkf = EnsembleKalmanFilter()
        qpens = QPEnsemble()
        obs_generator = ObservationGenerator(rngs)

        histories = ExperimentHistory(
            truth=[],
            enkf_analysis=[],
            qpens_analysis=[],
            cnn_analysis=[],
            enkf_background=[],
            qpens_background=[],
            cnn_background=[],
            seed=seed,
        )

        for i in range(num_inference_steps):
            truth_model.propagate()
            truth_state = truth_model.get_state()
            histories.truth.append(truth_state.copy())

            obs_data = obs_generator.generate_observations_with_locations(
                truth_state, experiment_config.num_ensemble_members
            )

            enkf_model.propagate()
            enkf_state = enkf_model.get_state()
            histories.enkf_background.append(enkf_state)

            enkf_assimilated = enkf.assimilate(
                enkf_state, obs_data.observation, obs_data.locations
            )
            enkf_model.assimilate(enkf_assimilated)
            histories.enkf_analysis.append(enkf_assimilated.copy())

            qpens_model.propagate()
            qpens_state = qpens_model.get_state()
            histories.qpens_background.append(qpens_state)

            qpens_assimilated = qpens.assimilate(
                qpens_state, obs_data.observation, obs_data.locations
            )
            qpens_model.assimilate(qpens_assimilated)
            histories.qpens_analysis.append(qpens_assimilated.copy())

            cnn_model.propagate()
            cnn_state = cnn_model.get_state()
            histories.cnn_background.append(cnn_state)

            cnn_enkf_assimilated = enkf.assimilate(
                cnn_state, obs_data.observation, obs_data.locations
            )

            cnn_corrected = self._apply_cnn_correction(
                cnn_enkf_assimilated, obs_data.locations
            )
            cnn_model.assimilate(cnn_corrected)

            histories.cnn_analysis.append(cnn_corrected.copy())

        return histories

    def _apply_cnn_correction(
        self, assimilated_state: np.ndarray, observation_locations: np.ndarray
    ) -> np.ndarray:
        observation_locations_data = np.tile(
            np.expand_dims(observation_locations[2:3], axis=-1),
            (1, 1, assimilated_state.shape[2]),
        )

        state_with_obs = np.concatenate(
            [assimilated_state, observation_locations_data], axis=0
        )

        state_tensor = torch.tensor(
            state_with_obs, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        normalized_tensor = (state_tensor - self.norm_stats["mean_in"]) / (
            self.norm_stats["std_in"] + 1e-8
        )

        with torch.no_grad():
            corrected_norm = self.model(normalized_tensor)

        corrected_tensor = (
            corrected_norm * self.norm_stats["std_out"]
        ) + self.norm_stats["mean_out"]
        corrected_state = corrected_tensor.permute(1, 2, 0).cpu().numpy()

        return corrected_state


def run_pipeline(load_model_name: str = "") -> Tuple[List[ExperimentHistory], str]:
    pipeline = ExperimentPipeline(load_model_name)
    all_histories = []

    for i in range(experiment_config.num_seeds):
        seed = experiment_config.base_seed + i
        print(
            f"Running experiment {i + 1}/{experiment_config.num_seeds} with seed {seed}"
        )

        history = pipeline.run_single_experiment(
            seed, experiment_config.num_inference_steps
        )
        all_histories.append(history)

    return all_histories, pipeline.loaded_model_name


def save_histories(
    histories: List[ExperimentHistory],
    model_name: str,
):
    save_name = f"{model_name.removesuffix('.pth')}_{experiment_config.base_seed}_{experiment_config.num_seeds}.npz"
    save_path = f"{experiments_path}{save_name}"
    save_data = {}

    for i, hist in enumerate(histories):
        save_data[f"truth_{i}"] = np.array(hist.truth)
        save_data[f"enkf_analysis_{i}"] = np.array(hist.enkf_analysis)
        save_data[f"qpens_analysis_{i}"] = np.array(hist.qpens_analysis)
        save_data[f"cnn_analysis_{i}"] = np.array(hist.cnn_analysis)
        save_data[f"enkf_background_{i}"] = np.array(hist.enkf_background)
        save_data[f"qpens_background_{i}"] = np.array(hist.qpens_background)
        save_data[f"cnn_background_{i}"] = np.array(hist.cnn_background)
        save_data[f"seed_{i}"] = hist.seed

    save_data["num_experiments"] = len(histories)

    np.savez_compressed(save_path, **save_data)
    print(f"Histories saved to {save_path}")


def load_histories(load_name: str) -> List[ExperimentHistory]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = f"{experiments_path}{load_name}"
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    histories = []
    for i in range(num_experiments):
        history = ExperimentHistory(
            truth=list(data[f"truth_{i}"]),
            enkf_analysis=list(data[f"enkf_analysis_{i}"]),
            qpens_analysis=list(data[f"qpens_analysis_{i}"]),
            cnn_analysis=list(data[f"cnn_analysis_{i}"]),
            enkf_background=list(data[f"enkf_background_{i}"]),
            qpens_background=list(data[f"qpens_background_{i}"]),
            cnn_background=list(data[f"cnn_background_{i}"]),
            seed=int(data[f"seed_{i}"]),
        )
        histories.append(history)

    return histories


def generate_experiment_data(model_name: str = ""):
    histories, loaded_model_name = run_pipeline(model_name)
    save_histories(histories, loaded_model_name)

    uhr_rmse_comparison(histories)
    print(f"Generated {len(histories)} experiment histories")


def uhr_rmse_comparison(histories: List[ExperimentHistory]):
    """Create RMSE comparison plots for background and analysis states"""
    num_experiments = len(histories)
    num_steps = len(histories[0].truth)

    variables = ["u", "h", "r"]
    methods = ["enkf", "qpens", "cnn"]

    rmse_background = {method: {var: [] for var in variables} for method in methods}
    rmse_analysis = {method: {var: [] for var in variables} for method in methods}

    for hist in histories:
        truth_array = np.array(hist.truth)

        enkf_bg = np.array(hist.enkf_background)
        qpens_bg = np.array(hist.qpens_background)
        cnn_bg = np.array(hist.cnn_background)

        enkf_an = np.array(hist.enkf_analysis)
        qpens_an = np.array(hist.qpens_analysis)
        cnn_an = np.array(hist.cnn_analysis)

        for t in range(num_steps):
            truth_t = truth_array[t]

            for i, (method_bg, method_name) in enumerate(
                [(enkf_bg, "enkf"), (qpens_bg, "qpens"), (cnn_bg, "cnn")]
            ):
                if t < len(method_bg):
                    bg_t = method_bg[t]
                    bg_mean = np.mean(bg_t, axis=2, keepdims=True)

                    for var_idx, var_name in enumerate(variables):
                        rmse_val = np.sqrt(
                            np.mean((truth_t[var_idx] - bg_mean[var_idx]) ** 2)
                        )
                        if len(rmse_background[method_name][var_name]) <= t:
                            rmse_background[method_name][var_name].append([])
                        rmse_background[method_name][var_name][t].append(rmse_val)

            for i, (method_an, method_name) in enumerate(
                [(enkf_an, "enkf"), (qpens_an, "qpens"), (cnn_an, "cnn")]
            ):
                if t < len(method_an):
                    an_t = method_an[t]
                    an_mean = np.mean(an_t, axis=2, keepdims=True)

                    for var_idx, var_name in enumerate(variables):
                        rmse_val = np.sqrt(
                            np.mean((truth_t[var_idx] - an_mean[var_idx]) ** 2)
                        )
                        if len(rmse_analysis[method_name][var_name]) <= t:
                            rmse_analysis[method_name][var_name].append([])
                        rmse_analysis[method_name][var_name][t].append(rmse_val)

    rmse_bg_mean = {method: {var: [] for var in variables} for method in methods}
    rmse_an_mean = {method: {var: [] for var in variables} for method in methods}

    for method in methods:
        for var in variables:
            for t in range(num_steps):
                if (
                    t < len(rmse_background[method][var])
                    and rmse_background[method][var][t]
                ):
                    rmse_bg_mean[method][var].append(
                        np.mean(rmse_background[method][var][t])
                    )
                if (
                    t < len(rmse_analysis[method][var])
                    and rmse_analysis[method][var][t]
                ):
                    rmse_an_mean[method][var].append(
                        np.mean(rmse_analysis[method][var][t])
                    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("RMSE Comparison: Background vs Analysis", fontsize=16)

    colors = {"enkf": "blue", "qpens": "red", "cnn": "green"}
    labels = {"enkf": "EnKF", "qpens": "QPEns", "cnn": "NN"}

    for var_idx, var in enumerate(variables):
        ax = axes[0, var_idx]

        for method in methods:
            if rmse_bg_mean[method][var]:
                time_steps = range(len(rmse_bg_mean[method][var]))
                ax.plot(
                    time_steps,
                    rmse_bg_mean[method][var],
                    color=colors[method],
                    label=labels[method],
                    linewidth=2,
                )

        ax.set_title(f"RMSE {var}")
        ax.set_ylabel("Background")
        ax.grid(True, alpha=0.3)
        if var_idx == 2:
            ax.legend()

    for var_idx, var in enumerate(variables):
        ax = axes[1, var_idx]

        for method in methods:
            if rmse_an_mean[method][var]:
                time_steps = range(len(rmse_an_mean[method][var]))
                ax.plot(
                    time_steps,
                    rmse_an_mean[method][var],
                    color=colors[method],
                    label=labels[method],
                    linewidth=2,
                )

        ax.set_ylabel("Analysis")
        ax.set_xlabel("Time Steps")
        ax.grid(True, alpha=0.3)

    for var_idx in range(3):
        bg_ax = axes[0, var_idx]
        an_ax = axes[1, var_idx]

        bg_ylim = bg_ax.get_ylim()
        an_ylim = an_ax.get_ylim()

        y_min = min(bg_ylim[0], an_ylim[0])
        y_max = max(bg_ylim[1], an_ylim[1])

        bg_ax.set_ylim(y_min, y_max)
        an_ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(
        f"{global_config.out_path}{global_config.animation_out_filename}/rmse_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    generate_experiment_data()
