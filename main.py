import os

from typer import Typer

from random_manager import RandomGenerators
from msw_model import ModifiedShallowWaterModel
from settings import load_settings
from assimilation import kf_assimilate, generate_observation, generate_radar_masks

app = Typer()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)
os.makedirs(settings.global_config.out_path, exist_ok=True)


@app.command()
def main():
    pass


@app.command()
def initialize_msw_model(ensemble_members: int = 10, save=True):
    """initializes the shallow water models"""
    state_truth = ModifiedShallowWaterModel(
        num_ensemble_members=1, random_generator=rngs.truth_rng
    )
    state_ensemble = ModifiedShallowWaterModel(
        num_ensemble_members=ensemble_members, random_generator=rngs.ensemble_rng
    )
    state_truth.initialize()
    state_ensemble.initialize()
    if save:
        state_truth.save_current_model_state()
        state_ensemble.save_current_model_state()

    return state_truth, state_ensemble


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


@app.command()
def assimilation(num_ensemble_members: int):
    model_truth, model_ensemble = initialize_msw_model(
        ensemble_members=num_ensemble_members, save=False
    )

    state_truth = model_truth.get_current_state()
    state_ensemble = model_ensemble.get_current_state()

    observation_ensemble = generate_observation(
        truth_state=state_truth,
        num_ensemble_members=num_ensemble_members,
        random_generator=rngs.obs_rng,
    )
    observation_locations = generate_radar_masks(
        state_truth=state_truth, random_generator=rngs.radar_rng
    )

    state_assimilated = kf_assimilate(
        ensemble=state_ensemble,
        observation=observation_ensemble,
        observation_position=observation_locations,
    )


if __name__ == "__main__":
    assimilation(10)
