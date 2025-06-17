import os

from typer import Typer

from random_manager import RandomGenerators
from msw_model import ModifiedShallowWaterModel
from settings import load_settings

app = Typer()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)
os.makedirs(settings.global_config.out_path, exist_ok=True)


@app.command()
def main():
    pass


@app.command()
def initialize_msw_model(ensemble_members: int = 10):
    """initializes the shallow water models"""
    state_truth = ModifiedShallowWaterModel(
        num_ensemble_members=1, random_generator=rngs.truth_rng
    )
    state_ensemble = ModifiedShallowWaterModel(
        num_ensemble_members=ensemble_members, random_generator=rngs.ensemble_rng
    )
    state_truth.initialize()
    state_ensemble.initialize()
    state_truth.save_current_model_state()
    state_ensemble.save_current_model_state()


@app.command()
def visualize_existing_model_state_history(model_name: str):
    load_path = f"{settings.global_config.out_path}/{model_name}"
    state = ModifiedShallowWaterModel.from_state_history(load_path=load_path)
    state.animate_evolution()


@app.command()
def train_nn():
    pass


@app.command()
def assimilate():
    pass


if __name__ == "__main__":
    app()
