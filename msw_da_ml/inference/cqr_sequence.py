import os
import copy
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

from msw_da_ml.data.evaluation_sequences import load_evaluation_sequences
from msw_da_ml.inference.base_sequence import (
    as_float32,
    closed_loop_context_from_base,
    iter_observations_from_base,
    rain_unobserved_channel,
)
from msw_da_ml.inference.cqr_model_io import (
    load_cqr_trained_model,
)
from msw_da_ml.core.msw_model import EnsembleModel
from msw_da_ml.core.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.core.observations import ObservationGenerator
from msw_da_ml.core.random import RandomGenerators
from msw_da_ml.settings import get_evaluation_sequence_dir, load_settings

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

sequences_path = get_evaluation_sequence_dir("cqr")


@dataclass
class CqrEvaluationSequence:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis_lower_quantiles: List[np.ndarray]
    cnn_analysis_upper_quantiles: List[np.ndarray]
    seed: int


class CqrEvaluationSequenceGenerator:
    def __init__(self, load_model_name: str = ""):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.norm_stats = load_cqr_trained_model(
            load_model_name, self.device
        )

    def run_single_experiment(
        self, seed: int, num_inference_steps: int
    ) -> CqrEvaluationSequence:
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

        sequence = CqrEvaluationSequence(
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
            sequence.truth.append(truth_state.copy())

            obs_data = obs_generator.generate_observations_with_locations(
                truth_state, experiment_config.num_ensemble_members
            )

            enkf_model.propagate()
            enkf_state = enkf_model.get_state()

            enkf_assimilated = enkf.assimilate(
                enkf_state, obs_data.observation, obs_data.locations
            )
            enkf_model.assimilate(enkf_assimilated)
            sequence.enkf_analysis.append(enkf_assimilated.copy())

            qpens_model.propagate()
            qpens_state = qpens_model.get_state()

            qpens_assimilated = qpens.assimilate(
                qpens_state, obs_data.observation, obs_data.locations
            )
            qpens_model.assimilate(qpens_assimilated)
            sequence.qpens_analysis.append(qpens_assimilated.copy())

            cnn_model.propagate()
            cnn_state = cnn_model.get_state()

            cnn_enkf_assimilated = enkf.assimilate(
                cnn_state, obs_data.observation, obs_data.locations
            )

            cnn_lower_quantile, cnn_upper_quantile = self._apply_cnn_correction(
                cnn_enkf_assimilated, obs_data.locations
            )
            cnn_corrected = 0.5 * (cnn_lower_quantile + cnn_upper_quantile)
            cnn_model.assimilate(cnn_corrected)

            sequence.cnn_analysis_lower_quantiles.append(cnn_lower_quantile.copy())
            sequence.cnn_analysis_upper_quantiles.append(cnn_upper_quantile.copy())

        return sequence

    def _apply_cnn_correction(
        self, assimilated_state: np.ndarray, observation_locations: np.ndarray
    ) -> np.ndarray:
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
            lower_quantile, upper_quantile = self.model(normalized_tensor)

        corrected_lower_quantile = (
            lower_quantile * self.norm_stats["std_out"]
        ) + self.norm_stats["mean_out"]
        corrected_upper_quantile = (
            upper_quantile * self.norm_stats["std_out"]
        ) + self.norm_stats["mean_out"]
        corrected_lower_quantile_permuted = (
            corrected_lower_quantile.permute(1, 2, 0).cpu().numpy()
        )
        corrected_upper_quantile_permuted = (
            corrected_upper_quantile.permute(1, 2, 0).cpu().numpy()
        )

        return corrected_lower_quantile_permuted, corrected_upper_quantile_permuted


def generate_cqr_evaluation_sequences(
    load_model_name: str = "",
) -> Tuple[List[CqrEvaluationSequence], str]:
    generator = CqrEvaluationSequenceGenerator(load_model_name)
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

    return sequences, load_model_name


def save_cqr_evaluation_sequences(
    sequences: List[CqrEvaluationSequence],
    loaded_model_name: str,
):
    model_stem = Path(loaded_model_name).stem if loaded_model_name else "latest"
    model_id = model_stem.removeprefix("cqr_")
    save_name = (
        f"eval_cqr_model-{model_id}"
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
        save_data[f"cnn_analysis_lower_quantiles_{i}"] = as_float32(
            sequence.cnn_analysis_lower_quantiles
        )
        save_data[f"cnn_analysis_upper_quantiles_{i}"] = as_float32(
            sequence.cnn_analysis_upper_quantiles
        )
        save_data[f"seed_{i}"] = sequence.seed

    save_data["num_experiments"] = len(sequences)

    np.savez_compressed(save_path, **save_data)
    print(f"CQR evaluation sequences saved to {save_path}")
    return save_name


def load_cqr_evaluation_sequences(load_name: str) -> List[CqrEvaluationSequence]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = os.path.join(sequences_path, load_name)
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    sequences = []
    for i in range(num_experiments):
        sequence = CqrEvaluationSequence(
            truth=list(data[f"truth_{i}"]),
            enkf_analysis=list(data[f"enkf_analysis_{i}"]),
            qpens_analysis=list(data[f"qpens_analysis_{i}"]),
            cnn_analysis_lower_quantiles=list(
                data[f"cnn_analysis_lower_quantiles_{i}"]
            ),
            cnn_analysis_upper_quantiles=list(
                data[f"cnn_analysis_upper_quantiles_{i}"]
            ),
            seed=int(data[f"seed_{i}"]),
        )
        sequences.append(sequence)

    return sequences


def generate_cqr_evaluation_sequences_from_base(
    base_sequence_name: str,
    load_model_name: str = "",
) -> Tuple[List[CqrEvaluationSequence], str]:
    generator = CqrEvaluationSequenceGenerator(load_model_name)
    base_sequences = load_evaluation_sequences(base_sequence_name)
    sequences = []

    for i, base_sequence in enumerate(base_sequences):
        print(
            f"Generating CQR predictions for base sequence {i + 1}/{len(base_sequences)} "
            f"with seed {base_sequence.seed}"
        )
        sequence = CqrEvaluationSequence(
            truth=list(base_sequence.truth),
            enkf_analysis=list(base_sequence.enkf_analysis),
            qpens_analysis=list(base_sequence.qpens_analysis),
            cnn_analysis_lower_quantiles=[],
            cnn_analysis_upper_quantiles=[],
            seed=base_sequence.seed,
        )

        context = closed_loop_context_from_base(base_sequence)
        for obs_data in iter_observations_from_base(
            base_sequence, context.observation_generator
        ):
            context.model.propagate()
            cnn_state = context.model.get_state()
            cnn_enkf_analysis = context.enkf.assimilate(
                cnn_state, obs_data.observation, obs_data.locations
            )
            lower_quantile, upper_quantile = generator._apply_cnn_correction(
                cnn_enkf_analysis, obs_data.locations
            )
            context.model.assimilate(0.5 * (lower_quantile + upper_quantile))
            sequence.cnn_analysis_lower_quantiles.append(lower_quantile.copy())
            sequence.cnn_analysis_upper_quantiles.append(upper_quantile.copy())

        sequences.append(sequence)

    return sequences, load_model_name


def generate_cqr_evaluation_data_from_base(
    base_sequence_name: str,
    model_name: str = "",
) -> str:
    sequences, loaded_model_name = generate_cqr_evaluation_sequences_from_base(
        base_sequence_name, model_name
    )
    save_name = save_cqr_evaluation_sequences(sequences, loaded_model_name)

    print(f"Generated {len(sequences)} CQR evaluation sequences from {base_sequence_name}")
    return save_name


def generate_cqr_evaluation_data(model_name: str = ""):
    sequences, loaded_model_name = generate_cqr_evaluation_sequences(model_name)
    save_name = save_cqr_evaluation_sequences(sequences, loaded_model_name)

    print(f"Generated {len(sequences)} CQR evaluation sequences")
    return save_name


if __name__ == "__main__":
    generate_cqr_evaluation_data()
