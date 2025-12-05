import os
import copy
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
from cyclopts import App

from msw_da_ml.msw_cnn.inference import load_trained_model
from msw_da_ml.msw.msw_model import EnsembleModel
from msw_da_ml.msw.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.msw.observation_generation import ObservationGenerator
from msw_da_ml.msw.random_manager import RandomGenerators
from msw_da_ml.settings import load_settings, get_output_dir
from msw_da_ml.evidential_regression.nig_network import NIGCNNModel


app = App()

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

experiments_path = get_output_dir(global_config.experiment_histories_out_filename)


@dataclass
class ExperimentHistoryNIG:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis_gamma: List[np.ndarray]
    cnn_analysis_nu: List[np.ndarray]
    cnn_analysis_alpha: List[np.ndarray]
    cnn_analysis_beta: List[np.ndarray]
    seed: int


class ExperimentPipelineNIG:
    def __init__(self, load_model_name: str = ""):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        name_begins_with = "nig"
        self.model, self.norm_stats = load_trained_model(
            NIGCNNModel, load_model_name, name_begins_with, self.device
        )
        self.model.eval()

    def run_single_experiment(
        self, seed: int, num_inference_steps: int
    ) -> ExperimentHistoryNIG:
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

        histories = ExperimentHistoryNIG(
            truth=[],
            enkf_analysis=[],
            qpens_analysis=[],
            cnn_analysis_gamma=[],
            cnn_analysis_nu=[],
            cnn_analysis_alpha=[],
            cnn_analysis_beta=[],
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

            gamma, nu, alpha, beta = self._apply_cnn_correction_nig(
                cnn_enkf_assimilated,
                obs_data.locations,
            )

            cnn_model.assimilate(gamma)

            histories.cnn_analysis_gamma.append(gamma.copy())
            histories.cnn_analysis_nu.append(nu.copy())
            histories.cnn_analysis_alpha.append(alpha.copy())
            histories.cnn_analysis_beta.append(beta.copy())

        return histories

    def _apply_cnn_correction_nig(
        self,
        assimilated_state: np.ndarray,
        observation_locations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            gamma, nu, alpha, beta = self.model(normalized_tensor)

        gamma_denorm = (
            gamma * self.norm_stats["std_out"]
        ) + self.norm_stats["mean_out"]

        gamma_np = gamma_denorm.permute(1, 2, 0).cpu().numpy()
        nu_np = nu.permute(1, 2, 0).cpu().numpy()
        alpha_np = alpha.permute(1, 2, 0).cpu().numpy()
        beta_np = beta.permute(1, 2, 0).cpu().numpy()

        return gamma_np, nu_np, alpha_np, beta_np


def run_pipeline(load_model_name: str = "") -> List[ExperimentHistoryNIG]:
    pipeline = ExperimentPipelineNIG(load_model_name)
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
    histories: List[ExperimentHistoryNIG],
):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    save_name = f"nig_hist_{timestamp}_{experiment_config.base_seed}_{experiment_config.num_seeds}.npz"
    save_path = os.path.join(experiments_path, save_name)
    save_data = {}

    for i, hist in enumerate(histories):
        save_data[f"truth_{i}"] = np.array(hist.truth)
        save_data[f"enkf_analysis_{i}"] = np.array(hist.enkf_analysis)
        save_data[f"qpens_analysis_{i}"] = np.array(hist.qpens_analysis)
        save_data[f"cnn_analysis_gamma_{i}"] = np.array(hist.cnn_analysis_gamma)
        save_data[f"cnn_analysis_nu_{i}"] = np.array(hist.cnn_analysis_nu)
        save_data[f"cnn_analysis_alpha_{i}"] = np.array(hist.cnn_analysis_alpha)
        save_data[f"cnn_analysis_beta_{i}"] = np.array(hist.cnn_analysis_beta)
        save_data[f"seed_{i}"] = hist.seed

    save_data["num_experiments"] = len(histories)

    np.savez_compressed(save_path, **save_data)
    print(f"Histories saved to {save_path}")


def load_histories(load_name: str) -> List[ExperimentHistoryNIG]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = os.path.join(experiments_path, load_name)
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    histories = []
    for i in range(num_experiments):
        history = ExperimentHistoryNIG(
            truth=list(data[f"truth_{i}"]),
            enkf_analysis=list(data[f"enkf_analysis_{i}"]),
            qpens_analysis=list(data[f"qpens_analysis_{i}"]),
            cnn_analysis_gamma=list(data[f"cnn_analysis_gamma_{i}"]),
            cnn_analysis_nu=list(data[f"cnn_analysis_nu_{i}"]),
            cnn_analysis_alpha=list(data[f"cnn_analysis_alpha_{i}"]),
            cnn_analysis_beta=list(data[f"cnn_analysis_beta_{i}"]),
            seed=int(data[f"seed_{i}"]),
        )
        histories.append(history)

    return histories


@app.command()
def generate_experiment_data(model_name: str = ""):
    histories = run_pipeline(model_name)
    save_histories(histories)

    print(f"Generated {len(histories)} experiment histories")


if __name__ == "__main__":
    app()