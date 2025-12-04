import os
import copy
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np

from msw_da_ml.msw_cnn.inference import load_trained_model
from msw_da_ml.msw.msw_model import EnsembleModel
from msw_da_ml.msw.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.msw.observation_generation import ObservationGenerator
from msw_da_ml.msw.random_manager import RandomGenerators
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

experiments_path = get_output_dir(global_config.experiment_histories_out_filename)


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
        self.model, self.norm_stats = load_trained_model(
            load_model_name=load_model_name,
            name_begins_with="model",
            device=self.device,
        )

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

    return all_histories


def save_histories(
    histories: List[ExperimentHistory],
):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    save_name = f"hist_{timestamp}_{experiment_config.base_seed}_{experiment_config.num_seeds}.npz"
    save_path = os.path.join(experiments_path, save_name)
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

    load_path = os.path.join(experiments_path, load_name)
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
    histories = run_pipeline(model_name)
    save_histories(histories)

    print(f"Generated {len(histories)} experiment histories")


if __name__ == "__main__":
    generate_experiment_data()
