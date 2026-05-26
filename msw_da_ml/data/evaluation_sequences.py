import os
import copy
from typing import List
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

from msw_da_ml.inference.cnn_sequence import load_trained_model
from msw_da_ml.core.msw_model import EnsembleModel
from msw_da_ml.core.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.core.observations import ObservationGenerator
from msw_da_ml.core.random import RandomGenerators
from msw_da_ml.settings import get_evaluation_sequence_dir, load_settings

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

sequences_path = get_evaluation_sequence_dir("cnn")


@dataclass
class EvaluationSequence:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis: List[np.ndarray]
    enkf_background: List[np.ndarray]
    qpens_background: List[np.ndarray]
    cnn_background: List[np.ndarray]
    cnn_enkf_analysis: List[np.ndarray]
    observation_locations: List[np.ndarray]
    seed: int


class EvaluationSequenceGenerator:
    def __init__(self, load_model_name: str = ""):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.norm_stats = load_trained_model(
            load_model_name=load_model_name,
            name_begins_with="cnn",
            device=self.device,
        )

    def run_single_experiment(
        self, seed: int, num_inference_steps: int
    ) -> EvaluationSequence:
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

        sequence = EvaluationSequence(
            truth=[],
            enkf_analysis=[],
            qpens_analysis=[],
            cnn_analysis=[],
            enkf_background=[],
            qpens_background=[],
            cnn_background=[],
            cnn_enkf_analysis=[],
            observation_locations=[],
            seed=seed,
        )

        for i in range(num_inference_steps):
            truth_model.propagate()
            truth_state = truth_model.get_state()
            sequence.truth.append(truth_state.copy())

            obs_data = obs_generator.generate_observations_with_locations(
                truth_state, experiment_config.num_ensemble_members
            )
            sequence.observation_locations.append(obs_data.locations.copy())

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
            sequence.cnn_enkf_analysis.append(cnn_enkf_assimilated.copy())

            cnn_corrected = self._apply_cnn_correction(
                cnn_enkf_assimilated, obs_data.locations
            )
            cnn_model.assimilate(cnn_corrected)

            sequence.cnn_analysis.append(cnn_corrected.copy())

        return sequence

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


def generate_evaluation_sequences(load_model_name: str = "") -> List[EvaluationSequence]:
    generator = EvaluationSequenceGenerator(load_model_name)
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


def _as_float32(values: List[np.ndarray]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _evaluation_sequence_name(method: str, model_name: str = "") -> str:
    model_stem = Path(model_name).stem if model_name else "latest"
    prefix = f"{method}_"
    model_id = model_stem.removeprefix(prefix)
    return (
        f"eval_{method}_model-{model_id}"
        f"_seed{experiment_config.base_seed}_S{experiment_config.num_seeds}"
        f"_T{experiment_config.num_inference_steps}"
        f"_E{experiment_config.num_ensemble_members}"
        f"_G{settings.water_model_config.ngrid}.npz"
    )


def save_evaluation_sequences(
    sequences: List[EvaluationSequence],
    model_name: str = "",
) -> str:
    save_name = _evaluation_sequence_name("cnn", model_name)
    save_path = os.path.join(sequences_path, save_name)
    save_data = {}

    for i, sequence in enumerate(sequences):
        save_data[f"truth_{i}"] = _as_float32(sequence.truth)
        save_data[f"enkf_analysis_{i}"] = _as_float32(sequence.enkf_analysis)
        save_data[f"qpens_analysis_{i}"] = _as_float32(sequence.qpens_analysis)
        save_data[f"cnn_analysis_{i}"] = _as_float32(sequence.cnn_analysis)
        save_data[f"enkf_background_{i}"] = _as_float32(sequence.enkf_background)
        save_data[f"qpens_background_{i}"] = _as_float32(sequence.qpens_background)
        save_data[f"cnn_background_{i}"] = _as_float32(sequence.cnn_background)
        save_data[f"cnn_enkf_analysis_{i}"] = _as_float32(sequence.cnn_enkf_analysis)
        save_data[f"observation_locations_{i}"] = np.asarray(
            sequence.observation_locations, dtype=bool
        )
        save_data[f"seed_{i}"] = sequence.seed

    save_data["num_experiments"] = len(sequences)

    np.savez_compressed(save_path, **save_data)
    print(f"Evaluation sequences saved to {save_path}")
    return save_name


def _optional_sequence_field(data, key: str) -> list[np.ndarray]:
    if key not in data.files:
        return []
    return list(data[key])


def load_evaluation_sequences(load_name: str) -> List[EvaluationSequence]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = os.path.join(sequences_path, load_name)
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    sequences = []
    for i in range(num_experiments):
        sequence = EvaluationSequence(
            truth=list(data[f"truth_{i}"]),
            enkf_analysis=list(data[f"enkf_analysis_{i}"]),
            qpens_analysis=list(data[f"qpens_analysis_{i}"]),
            cnn_analysis=list(data[f"cnn_analysis_{i}"]),
            enkf_background=list(data[f"enkf_background_{i}"]),
            qpens_background=list(data[f"qpens_background_{i}"]),
            cnn_background=list(data[f"cnn_background_{i}"]),
            cnn_enkf_analysis=_optional_sequence_field(data, f"cnn_enkf_analysis_{i}"),
            observation_locations=_optional_sequence_field(
                data, f"observation_locations_{i}"
            ),
            seed=int(data[f"seed_{i}"]),
        )
        sequences.append(sequence)

    return sequences


def generate_evaluation_data(model_name: str = "") -> str:
    sequences = generate_evaluation_sequences(model_name)
    save_name = save_evaluation_sequences(sequences, model_name)

    print(f"Generated {len(sequences)} evaluation sequences")
    return save_name

if __name__ == "__main__":
    generate_evaluation_data()
