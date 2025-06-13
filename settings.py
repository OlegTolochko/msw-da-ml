from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class WaterModelConfig(BaseModel):
    num_grid_cells: int = Field(gt=0)
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


class AppSettings(BaseModel):
    water_model_config: WaterModelConfig


def load_settings(path: str = "config.yaml") -> AppSettings:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppSettings.model_validate(config_data)
