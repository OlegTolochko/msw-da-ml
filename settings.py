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


class GlobalConfig(BaseModel):
    out_path: str
    base_seed: int


class AppSettings(BaseModel):
    water_model_config: WaterModelConfig
    global_config: GlobalConfig


def load_settings(path: str = "config.yaml") -> AppSettings:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppSettings.model_validate(config_data)
