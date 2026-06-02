import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from msw_da_ml.data.evaluation_sequences import load_evaluation_sequences
from msw_da_ml.inference.base_sequence import (
    as_float32,
    closed_loop_context_from_base,
    iter_observations_from_base,
    rain_unobserved_channel,
)
from msw_da_ml.models.mcdo import MCDOCNNModel
from msw_da_ml.settings import (
    get_evaluation_sequence_dir,
    get_model_artifact_dir,
    get_model_artifact_paths,
    load_settings,
)


settings = load_settings()
experiment_config = settings.experiment_config
sequences_path = get_evaluation_sequence_dir("ensemble")


@dataclass
class EnsembleEvaluationSequence:
    truth: List[np.ndarray]
    enkf_analysis: List[np.ndarray]
    qpens_analysis: List[np.ndarray]
    cnn_analysis_mean: List[np.ndarray]
    cnn_analysis_logvar: List[np.ndarray]
    enkf_background: List[np.ndarray]
    qpens_background: List[np.ndarray]
    cnn_background: List[np.ndarray]
    seed: int


class DeepEnsembleEvaluationSequenceGenerator:
    def __init__(self, load_model_name: str = ""):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.models, self.norm_stats = load_deep_ensemble_model(
            load_model_name, self.device
        )

    def _apply_cnn_correction_ensemble(
        self,
        assimilated_state: np.ndarray,
        observation_locations: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

        corrections = []
        correction_logvars = []
        with torch.no_grad():
            for model in self.models:
                corrected_norm_mean, corrected_norm_logvar = model(normalized_tensor)
                corrected_tensor = (
                    corrected_norm_mean * self.norm_stats["std_out"]
                ) + self.norm_stats["mean_out"]
                corrected_tensor_logvar = corrected_norm_logvar + 2 * torch.log(
                    self.norm_stats["std_out"] + 1e-8
                )
                corrections.append(corrected_tensor.permute(1, 2, 0).cpu().numpy())
                correction_logvars.append(
                    corrected_tensor_logvar.permute(1, 2, 0).cpu().numpy()
                )

        return corrections, correction_logvars


def load_deep_ensemble_model(
    load_model_name: str,
    device: str,
) -> tuple[list[torch.nn.Module], dict[str, torch.Tensor]]:
    if not load_model_name:
        load_model_name = get_most_recent_ensemble_model_name()
        if not load_model_name:
            raise FileNotFoundError("No deep ensemble model files found")

    if not load_model_name.endswith(".pth"):
        load_model_name += ".pth"

    model_load_path, norm_stats_path = get_model_artifact_paths(load_model_name)
    checkpoint = torch.load(model_load_path, map_location=device)
    member_state_dicts = _load_member_state_dicts(checkpoint, model_load_path, device)

    models = []
    for state_dict in member_state_dicts:
        model = MCDOCNNModel(dropout=0.0)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        models.append(model)

    norm_stats = torch.load(norm_stats_path, map_location=device)
    print(f"Loaded {len(models)} deep ensemble members.")
    return models, norm_stats


def get_most_recent_ensemble_model_name() -> str | None:
    most_recent_model = None
    most_recent_time = 0
    for model in os.scandir(get_model_artifact_dir("ensemble")):
        if not model.is_file() or not model.name.endswith(".pth"):
            continue
        if "_member" in Path(model.name).stem:
            continue
        mod_time = model.stat().st_mtime_ns
        if mod_time > most_recent_time:
            most_recent_model = model
            most_recent_time = mod_time
    return os.path.basename(most_recent_model.path) if most_recent_model else None


def _load_member_state_dicts(
    checkpoint,
    model_load_path: Path,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    if "member_model_files" in checkpoint:
        return [
            torch.load(model_load_path.with_name(member_file), map_location=device)
            for member_file in checkpoint["member_model_files"]
        ]
    return checkpoint["member_state_dicts"]


def generate_ensemble_evaluation_sequences_from_base(
    base_sequence_name: str,
    load_model_name: str = "",
) -> List[EnsembleEvaluationSequence]:
    generator = DeepEnsembleEvaluationSequenceGenerator(load_model_name)
    base_sequences = load_evaluation_sequences(base_sequence_name)
    sequences = []

    for i, base_sequence in enumerate(base_sequences):
        print(
            f"Generating ensemble predictions for base sequence "
            f"{i + 1}/{len(base_sequences)} with seed {base_sequence.seed}"
        )
        sequence = EnsembleEvaluationSequence(
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
            samples, logvars = generator._apply_cnn_correction_ensemble(
                cnn_enkf_analysis,
                obs_data.locations,
            )
            stacked_samples = np.stack(samples, axis=-1)
            context.model.assimilate(np.mean(stacked_samples, axis=-1))
            sequence.cnn_analysis_mean.append(stacked_samples.copy())
            sequence.cnn_analysis_logvar.append(np.stack(logvars, axis=-1).copy())

        sequences.append(sequence)

    return sequences


def save_ensemble_evaluation_sequences(
    sequences: List[EnsembleEvaluationSequence],
    model_name: str = "",
) -> str:
    model_stem = Path(model_name).stem if model_name else "latest"
    model_id = model_stem.removeprefix("ensemble_")
    save_name = (
        f"eval_ensemble_model-{model_id}"
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
    print(f"Deep ensemble evaluation sequences saved to {save_path}")
    return save_name


def load_ensemble_evaluation_sequences(
    load_name: str,
) -> List[EnsembleEvaluationSequence]:
    if not load_name.endswith(".npz"):
        load_name += ".npz"

    load_path = os.path.join(sequences_path, load_name)
    data = np.load(load_path)
    num_experiments = int(data["num_experiments"])

    sequences = []
    for i in range(num_experiments):
        sequence = EnsembleEvaluationSequence(
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


def generate_ensemble_evaluation_data_from_base(
    base_sequence_name: str,
    model_name: str = "",
) -> str:
    sequences = generate_ensemble_evaluation_sequences_from_base(
        base_sequence_name, model_name
    )
    save_name = save_ensemble_evaluation_sequences(sequences, model_name)

    print(
        f"Generated {len(sequences)} ensemble evaluation sequences "
        f"from {base_sequence_name}"
    )
    return save_name
