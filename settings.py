from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class WaterModelConfig(BaseModel):
    num_grid_cells: int = Field(gt=0)
    num_sub_steps: int


class AppSettings(BaseModel):
    water_model_config: WaterModelConfig


def load_settings(path: str = "config.yaml") -> AppSettings:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppSettings.model_validate(config_data)
