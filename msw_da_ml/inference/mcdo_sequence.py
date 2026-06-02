import os
import copy
from typing import List
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from cyclopts import App

from msw_da_ml.data.evaluation_sequences import load_evaluation_sequences
from msw_da_ml.inference.base_sequence import (
    as_float32,
    closed_loop_context_from_base,
    iter_observations_from_base,
    rain_unobserved_channel,
)
from msw_da_ml.inference.cnn_sequence import load_trained_model
from msw_da_ml.core.msw_model import EnsembleModel
from msw_da_ml.core.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.core.observations import ObservationGenerator
from msw_da_ml.core.random import RandomGenerators
from msw_da_ml.settings import get_evaluation_sequence_dir, load_settings
from msw_da_ml.models.mcdo import MCDOCNNModel


app = App()

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

sequences_path = get_evaluation_sequence_dir("mcdo")


@dataclass
class McdoEvaluationSequence:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis_mean: List[np.ndarray]
    cnn_analysis_logvar: List[np.ndarray]
    enkf_background: List[np.ndarray]
    qpens_background: List[np.ndarray]
    cnn_background: List[np.ndarray]
    seed: int


class McdoEvaluationSequenceGenerator:
    def __init__(self, load_model_name: str = ""):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        name_begins_with = "mcdo"
        self.model, self.norm_stats = load_trained_model(
            MCDOCNNModel, load_model_name, name_begins_with, self.device
        )
        self.model.train()  # set to train mode for MCDO

    def run_single_experiment(
        self, seed: int, num_inference_steps: int
    ) -> McdoEvaluationSequence:
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

        sequence = McdoEvaluationSequence(
            truth=[],
            enkf_analysis=[],
            qpens_analysis=[],
            cnn_analysis_mean=[],
            cnn_analysis_logvar=[],
            enkf_background=[],
            qpens_background=[],
            cnn_background=[],
            seed=seed,
        )

        for i in range(num_inference_steps):
            truth_model.propagate()
            truth_state = truth_model.get_state()
            sequence.truth.append(truth_state.copy())

            obs_data = obs_generator.generate_observations_with_locations(
                truth_state, experiment_config.num_ensemble_members
            )

            enkf_model.propagate()
            enkf_state = enkf_model.get_state()
            sequence.enkf_background.append(enkf_state)

            enkf_assimilated = enkf.assimilate(
                enkf_state, obs_data.observation, obs_data.locations
            )
            enkf_model.assimilate(enkf_assimilated)
            sequence.enkf_analysis.append(enkf_assimilated.copy())

            qpens_model.propagate()
            qpens_state = qpens_model.get_state()
            sequence.qpens_background.append(qpens_state)

            qpens_assimilated = qpens.assimilate(
                qpens_state, obs_data.observation, obs_data.locations
            )
            qpens_model.assimilate(qpens_assimilated)
            sequence.qpens_analysis.append(qpens_assimilated.copy())

            cnn_model.propagate()
            cnn_state = cnn_model.get_state()
            sequence.cnn_background.append(cnn_state)

            cnn_enkf_assimilated = enkf.assimilate(
                cnn_state, obs_data.observation, obs_data.locations
            )

            cnn_corrected_mcdo, cnn_corrected_logvars = self._apply_cnn_correction_mcdo(
                cnn_enkf_assimilated,
                obs_data.locations,
                experiment_config.num_ensemble_members,
            )

            # Stack MC dropout samples along the last axis:
            # (variables, gridpoints, ensemble_members, mc_samples)
            cnn_corrected_mcdo = np.stack(cnn_corrected_mcdo, axis=-1)

            cnn_corrected_mean = np.mean(
                cnn_corrected_mcdo, axis=-1
            )  # (3, gridpoints, num_ens_members)
            cnn_model.assimilate(cnn_corrected_mean)

            cnn_corrected_logvars = np.stack(cnn_corrected_logvars, axis=-1)
            sequence.cnn_analysis_mean.append(cnn_corrected_mcdo.copy())
            sequence.cnn_analysis_logvar.append(cnn_corrected_logvars.copy())

        return sequence

    def _apply_cnn_correction_mcdo(
        self,
        assimilated_state: np.ndarray,
        observation_locations: np.ndarray,
        num_iterations: int,
    ) -> np.ndarray:
        corrections = []
        correction_logvars = []
        for i in range(num_iterations):
            observation_locations_data = rain_unobserved_channel(
                observation_locations, assimilated_state.shape[2]
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
                corrected_norm_mean, corrected_norm_logvar = self.model(
                    normalized_tensor
                )

            corrected_tensor = (
                corrected_norm_mean * self.norm_stats["std_out"]
            ) + self.norm_stats["mean_out"]
            corrected_tensor_logvar = corrected_norm_logvar + 2 * torch.log(
                self.norm_stats["std_out"] + 1e-8
            )
            corrected_state = corrected_tensor.permute(1, 2, 0).cpu().numpy()
            corrected_state_logvars = (
                corrected_tensor_logvar.permute(1, 2, 0).cpu().numpy()
            )
            corrections.append(corrected_state)
            correction_logvars.append(corrected_state_logvars)

        return corrections, correction_logvars


def generate_mcdo_evaluation_sequences(load_model_name: str = "") -> List[McdoEvaluationSequence]:
    generator = McdoEvaluationSequenceGenerator(load_model_name)
    sequences = []

    for i in range(experiment_config.num_seeds):
        seed = experiment_config.base_seed + i
        print(
            f"Running experiment {i + 1}/{experiment_config.num_seeds} with seed {seed}"
        )

        sequence = generator.run_single_experiment(
            seed, experiment_config.num_inference_steps
        )
        sequences.append(sequence)

    return sequences


def save_mcdo_evaluation_sequences(
    sequences: List[McdoEvaluationSequence],
    model_name: str = "",
):
    model_stem = Path(model_name).stem if model_name else "latest"
    model_id = model_stem.removeprefix("mcdo_")
    save_name = (
        f"eval_mcdo_model-{model_id}"
        f"_seed{experiment_config.base_seed}_S{experiment_config.num_seeds}"
        f"_T{experiment_config.num_inference_steps}"
        f"_E{experiment_config.num_ensemble_members}"
        f"_G{settings.water_model_config.ngrid}.npz"
    )
    save_path = os.path.join(sequences_path, save_name)
    save_data = {}

    for i, sequence in enumerate(sequences):
        save_data[f"truth_{i}"] = as_float32(sequence.truth)
        save_data[f"enkf_analysis_{i}"] = as_float32(sequence.enkf_analysis)
        save_data[f"qpens_analysis_{i}"] = as_float32(sequence.qpens_analysis)
        save_data[f"cnn_analysis_mean_{i}"] = as_float32(sequence.cnn_analysis_mean)
        save_data[f"cnn_analysis_logvar_{i}"] = as_float32(sequence.cnn_analysis_logvar)
        save_data[f"enkf_background_{i}"] = as_float32(sequence.enkf_background)
        save_data[f"qpens_background_{i}"] = as_float32(sequence.qpens_background)
        save_data[f"cnn_background_{i}"] = as_float32(sequence.cnn_background)
        save_data[f"seed_{i}"] = sequence.seed

    save_data["num_experiments"] = len(sequences)

    np.savez_compressed(save_path, **save_data)
    print(f"MCDO evaluation sequences saved to {save_path}")
    return save_name


def load_mcdo_evaluation_sequences(load_name: str) -> List[McdoEvaluationSequence]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = os.path.join(sequences_path, load_name)
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    sequences = []
    for i in range(num_experiments):
        sequence = McdoEvaluationSequence(
            truth=list(data[f"truth_{i}"]),
            enkf_analysis=list(data[f"enkf_analysis_{i}"]),
            qpens_analysis=list(data[f"qpens_analysis_{i}"]),
            cnn_analysis_mean=list(data[f"cnn_analysis_mean_{i}"]),
            cnn_analysis_logvar=list(data[f"cnn_analysis_logvar_{i}"]),
            enkf_background=list(data[f"enkf_background_{i}"]),
            qpens_background=list(data[f"qpens_background_{i}"]),
            cnn_background=list(data[f"cnn_background_{i}"]),
            seed=int(data[f"seed_{i}"]),
        )
        sequences.append(sequence)

    return sequences


def generate_mcdo_evaluation_sequences_from_base(
    base_sequence_name: str,
    load_model_name: str = "",
) -> List[McdoEvaluationSequence]:
    generator = McdoEvaluationSequenceGenerator(load_model_name)
    base_sequences = load_evaluation_sequences(base_sequence_name)
    sequences = []

    for i, base_sequence in enumerate(base_sequences):
        print(
            f"Generating MCDO predictions for base sequence {i + 1}/{len(base_sequences)} "
            f"with seed {base_sequence.seed}"
        )
        sequence = McdoEvaluationSequence(
            truth=list(base_sequence.truth),
            enkf_analysis=list(base_sequence.enkf_analysis),
            qpens_analysis=list(base_sequence.qpens_analysis),
            cnn_analysis_mean=[],
            cnn_analysis_logvar=[],
            enkf_background=list(base_sequence.enkf_background),
            qpens_background=list(base_sequence.qpens_background),
            cnn_background=[],
            seed=base_sequence.seed,
        )

        context = closed_loop_context_from_base(base_sequence)
        for obs_data in iter_observations_from_base(
            base_sequence, context.observation_generator
        ):
            context.model.propagate()
            cnn_state = context.model.get_state()
            sequence.cnn_background.append(cnn_state.copy())

            cnn_enkf_analysis = context.enkf.assimilate(
                cnn_state, obs_data.observation, obs_data.locations
            )
            samples, logvars = generator._apply_cnn_correction_mcdo(
                cnn_enkf_analysis,
                obs_data.locations,
                experiment_config.num_ensemble_members,
            )
            stacked_samples = np.stack(samples, axis=-1)
            context.model.assimilate(np.mean(stacked_samples, axis=-1))
            sequence.cnn_analysis_mean.append(stacked_samples.copy())
            sequence.cnn_analysis_logvar.append(np.stack(logvars, axis=-1).copy())

        sequences.append(sequence)

    return sequences


def generate_mcdo_evaluation_data_from_base(
    base_sequence_name: str,
    model_name: str = "",
) -> str:
    sequences = generate_mcdo_evaluation_sequences_from_base(
        base_sequence_name, model_name
    )
    save_name = save_mcdo_evaluation_sequences(sequences, model_name)

    print(f"Generated {len(sequences)} MCDO evaluation sequences from {base_sequence_name}")
    return save_name


@app.command()
def generate_mcdo_evaluation_data(model_name: str = ""):
    sequences = generate_mcdo_evaluation_sequences(model_name)
    save_name = save_mcdo_evaluation_sequences(sequences, model_name)

    print(f"Generated {len(sequences)} MCDO evaluation sequences")
    return save_name


if __name__ == "__main__":
    app()
