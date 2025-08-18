import os
import sys
import copy
from typing import List, Tuple
from dataclasses import dataclass
import datetime

import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantile_inference import load_normalization, load_trained_model
from models.msw_model import EnsembleModel
from assimilation.assimilation import EnsembleKalmanFilter, QPEnsemble
from data.observation_generation import ObservationGenerator
from core.random_manager import RandomGenerators
from core.settings import load_settings

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

experiments_path = (
    f"{global_config.out_path}{global_config.quantile_experiment_histories_out_filename}"
)
os.makedirs(experiments_path, exist_ok=True)


@dataclass
class QuantileExperimentHistory:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis_lower_quantiles: List[np.ndarray]
    cnn_analysis_upper_quantiles: List[np.ndarray]
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
    ) -> QuantileExperimentHistory:
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

        histories = QuantileExperimentHistory(
            truth=[],
            enkf_analysis=[],
            qpens_analysis=[],
            cnn_analysis_lower_quantiles=[],
            cnn_analysis_upper_quantiles=[],
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

            enkf_assimilated = enkf.assimilate(
                enkf_state, obs_data.observation, obs_data.locations
            )
            enkf_model.assimilate(enkf_assimilated)
            histories.enkf_analysis.append(enkf_assimilated.copy())

            qpens_model.propagate()
            qpens_state = qpens_model.get_state()

            qpens_assimilated = qpens.assimilate(
                qpens_state, obs_data.observation, obs_data.locations
            )
            qpens_model.assimilate(qpens_assimilated)
            histories.qpens_analysis.append(qpens_assimilated.copy())

            cnn_model.propagate()
            cnn_state = cnn_model.get_state()

            cnn_enkf_assimilated = enkf.assimilate(
                cnn_state, obs_data.observation, obs_data.locations
            )

            cnn_lower_quantile, cnn_upper_quantile = self._apply_cnn_correction(
                cnn_enkf_assimilated, obs_data.locations
            )
            cnn_corrected = cnn_lower_quantile + (cnn_upper_quantile-cnn_lower_quantile)
            cnn_model.assimilate(cnn_corrected)

            histories.cnn_analysis_lower_quantiles.append(cnn_lower_quantile.copy())
            histories.cnn_analysis_upper_quantiles.append(cnn_upper_quantile.copy())

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
            lower_quantile, upper_quantile = self.model(normalized_tensor)

        corrected_lower_quantile = (
            lower_quantile * self.norm_stats["std_out"]
        ) + self.norm_stats["mean_out"]
        corrected_upper_quantile =  (
            upper_quantile * self.norm_stats["std_out"]
        ) + self.norm_stats["mean_out"]
        corrected_lower_quantile_permuted = corrected_lower_quantile.permute(1, 2, 0).cpu().numpy()
        corrected_upper_quantile_permuted = corrected_upper_quantile.permute(1, 2, 0).cpu().numpy()

        return corrected_lower_quantile_permuted, corrected_upper_quantile_permuted


def run_pipeline(load_model_name: str = "") -> Tuple[List[QuantileExperimentHistory], str]:
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
    histories: List[QuantileExperimentHistory],
    loaded_model_name: str,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    save_name = f"quantile_hist_{loaded_model_name}_{timestamp}_{experiment_config.base_seed}_{experiment_config.num_seeds}.npz"
    save_path = f"{experiments_path}{save_name}"
    save_data = {}

    for i, hist in enumerate(histories):
        save_data[f"truth_{i}"] = np.array(hist.truth)
        save_data[f"enkf_analysis_{i}"] = np.array(hist.enkf_analysis)
        save_data[f"qpens_analysis_{i}"] = np.array(hist.qpens_analysis)
        save_data[f"cnn_analysis_lower_quantiles_{i}"] = np.array(hist.cnn_analysis_lower_quantiles)
        save_data[f"cnn_analysis_upper_quantiles_{i}"] = np.array(hist.cnn_analysis_upper_quantiles)
        save_data[f"seed_{i}"] = hist.seed

    save_data["num_experiments"] = len(histories)

    np.savez_compressed(save_path, **save_data)
    print(f"Quantile histories saved to {save_path}")
    return save_name


def load_histories(load_name: str) -> List[QuantileExperimentHistory]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = f"{experiments_path}{load_name}"
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    histories = []
    for i in range(num_experiments):
        history = QuantileExperimentHistory(
            truth=list(data[f"truth_{i}"]),
            enkf_analysis=list(data[f"enkf_analysis_{i}"]),
            qpens_analysis=list(data[f"qpens_analysis_{i}"]),
            cnn_analysis_lower_quantiles=list(data[f"cnn_analysis_lower_quantiles_{i}"]),
            cnn_analysis_upper_quantiles=list(data[f"cnn_analysis_upper_quantiles_{i}"]),
            seed=int(data[f"seed_{i}"]),
        )
        histories.append(history)

    return histories


def generate_experiment_data(model_name: str = ""):
    histories, loaded_model_name = run_pipeline(model_name)
    save_name = save_histories(histories, loaded_model_name)

    print(f"Generated {len(histories)} quantile experiment histories")
    return save_name

generate_experiment_data()