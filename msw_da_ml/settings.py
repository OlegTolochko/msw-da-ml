from pydantic import BaseModel, Field
import yaml
from pathlib import Path


class WaterModelConfig(BaseModel):
    ngrid: int = Field(gt=0)
    num_sub_steps: int

    base_velocity: float
    base_height: float
    base_rain: float

    time_step_size: float
    grid_spacing: float

    h_cloud: float
    h_rain: float
    phi_cloud: float
    gravitational_constant: float
    r_gamma: float
    u_diff_coef: float
    h_diff_coef: float
    r_diff_coef: float
    r_removal_rate: float
    r_rate: float
    filter_coeff: float
    filter_correction_past: float
    filter_correction_present: float

    wind_perturbation_standard_deviation: float
    wind_perturbation_noise_amplitude: float


class SharedObsGenAssimilationConfig(BaseModel):
    u_error_std: float
    u_error_mean: float
    h_error_std: float
    h_error_mean: float
    r_error_std: float
    r_error_mean: float

    radar_rain_threshold: float
    radar_no_rain_observation_percentage: float
    cov_inflation: float
    grid_point_influence: int


class NetworkConfig(BaseModel):
    in_channels: int
    hidden_channels: int
    num_layers: int
    kernel_size: int


class LossConfig(BaseModel):
    variable_to_punish_idx: int
    bias_loss_weight: float


class TrainingConfig(BaseModel):
    val_split_size: float
    random_state_train_test_split: int
    batch_size: int
    learning_rate: float
    epochs: int
    mcdo_dropout: float
    deep_ensemble_size: int = 10
    spinup_cycles: int = 20


class InferenceConfig(BaseModel):
    inference_seed: int
    num_ensemble_members: int


class ExperimentConfig(BaseModel):
    base_seed: int
    num_seeds: int
    num_inference_steps: int
    num_ensemble_members: int
    spinup_cycles: int = 20


class ConformalPredictionConfig(BaseModel):
    calibration_split_ratio: float
    calibration_split_seed: int
    calibration_quantile: float
    rain_normalization_eps: float


class GlobalConfig(BaseModel):
    out_path: str
    visualizations_out_filename: str
    generated_data_animations_out_filename: str
    training_sequences_out_filename: str
    trained_nn_model_out_filename: str
    normalization_out_filename: str
    evaluation_sequences_out_filename: str
    trained_quantile_nn_model_out_filename: str
    quantile_normalization_out_filename: str
    quantile_evaluation_sequences_out_filename: str
    base_seed: int


class AppSettings(BaseModel):
    water_model_config: WaterModelConfig
    shared_obs_gen_assimilation_config: SharedObsGenAssimilationConfig
    network_config: NetworkConfig
    loss_config: LossConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    experiment_config: ExperimentConfig
    conformal_prediction_config: ConformalPredictionConfig
    global_config: GlobalConfig


def load_settings(path: str = None) -> AppSettings:
    if path is None:
        settings_dir = Path(__file__).parent
        path = settings_dir / "config.yaml"
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppSettings.model_validate(config_data)


def get_output_dir(subdir: str = "") -> Path:
    """Get the absolute output directory path.

    Args:
        subdir: Optional subdirectory within the output directory

    Returns:
        Path object pointing to the output directory
    """
    current_file = Path(__file__)
    # The location may be adjusted if wanted
    project_root = current_file.parent.parent

    settings = load_settings()
    base_out = Path(settings.global_config.out_path)

    if not base_out.is_absolute():
        base_out = project_root / base_out

    if subdir:
        base_out = base_out / subdir

    base_out.mkdir(parents=True, exist_ok=True)

    return base_out


def infer_model_family(name: str) -> str:
    """Infer the artifact family from a model file/name prefix."""
    stem = Path(name).stem.lower()
    if stem.startswith(("cqr", "quantile")):
        return "cqr"
    if stem.startswith("mcdo"):
        return "mcdo"
    if stem.startswith("evidential"):
        return "evidential"
    if stem.startswith(("ensemble", "deep_ensemble")):
        return "ensemble"
    if stem.startswith("rf"):
        return "rf"
    return "cnn"


def get_model_artifact_dir(name_or_family: str) -> Path:
    """Return the model artifact directory for a model family/name."""
    family = infer_model_family(name_or_family)
    return get_output_dir(f"models/{family}")


def get_model_artifact_paths(model_name: str) -> tuple[Path, Path]:
    """Return paired model and normalization-stat paths for a model name."""
    model_path = get_model_artifact_dir(model_name) / model_name
    if model_path.suffix != ".pth":
        model_path = model_path.with_suffix(".pth")
    norm_path = model_path.with_name(f"norm_{model_path.stem}.pt")
    return model_path, norm_path


def get_evaluation_sequence_dir(method: str) -> Path:
    """Return the evaluation sequence directory for a method."""
    return get_output_dir(f"data/evaluation/{method}")
