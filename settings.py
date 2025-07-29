from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


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
    bias_loss_weight: int


class TrainingConfig(BaseModel):
    val_split_size: float
    random_state_train_test_split: int
    batch_size: int
    learning_rate: float
    epochs: int
    model_save_name: str


class InferenceConfig(BaseModel):
    inference_seed: int
    num_ensemble_members: int


class ExperimentConfig(BaseModel):
    base_seed: int
    num_seeds: int
    num_inference_steps: int
    num_ensemble_members: int


class GlobalConfig(BaseModel):
    out_path: str
    animation_out_filename: str
    msw_model_out_filename: str
    trained_nn_model_out_filename: str
    normalization_out_filename: str
    experiment_histories_out_filename: str
    base_seed: int


class AppSettings(BaseModel):
    water_model_config: WaterModelConfig
    shared_obs_gen_assimilation_config: SharedObsGenAssimilationConfig
    network_config: NetworkConfig
    loss_config: LossConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    experiment_config: ExperimentConfig
    global_config: GlobalConfig


def load_settings(path: str = "config.yaml") -> AppSettings:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppSettings.model_validate(config_data)
