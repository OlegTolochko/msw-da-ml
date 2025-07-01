import os

from typer import Typer

from random_manager import RandomGenerators
from msw_model import ModifiedShallowWaterModel
from settings import load_settings
from assimilation import (
    kf_assimilate,
    qpens_assimilate,
    generate_observation,
    generate_radar_masks,
)

app = Typer()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)
os.makedirs(settings.global_config.out_path, exist_ok=True)


@app.command()
def main():
    pass


@app.command()
def visualize_existing_model_state_history(model_name: str):
    load_path = f"{settings.global_config.out_path}/{model_name}"
    state = ModifiedShallowWaterModel.from_state_history(load_path=load_path)
    state.animate_evolution()


@app.command()
def generate_observations_from_truth():
    pass


@app.command()
def train_nn():
    pass


if __name__ == "__main__":
    app()
