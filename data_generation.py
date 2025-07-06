import os
import copy

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
def generate_training_data(
    num_ensemble_members: int = 10,
    num_generation_steps: int = 20,
):
    """Three step process:
    1. forecast
    2. observation
    3. correction
    """
    model_truth, model_ensemble = initialize_msw_model(
        ensemble_members=num_ensemble_members, save=False
    )
    model_ensemble_kf = copy.deepcopy(model_ensemble)
    model_ensemble_qp = copy.deepcopy(model_ensemble)

    qp_state_history = []
    kf_state_history = []

    for i in range(num_generation_steps):
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

        state_assimilated_qp = qpens_assimilate(
            ensemble=state_ensemble,
            observation=observation_ensemble,
            observation_position=observation_locations,
        )
        qp_state_history.append(state_assimilated_qp)

        state_assimilated_kf = kf_assimilate(
            ensemble=state_ensemble,
            observation=observation_ensemble,
            observation_position=observation_locations,
        )
        kf_state_history.append(state_assimilated_kf)

        state_truth = model_truth.apply_nsub_steps(state_truth)

        model_ensemble_kf.apply_nsub_steps(state_assimilated_kf)
        model_ensemble_qp.apply_nsub_steps(state_assimilated_qp)

    model_ensemble_kf.animate_evolution_from_history(
        kf_state_history, "./out/model_evolution_kf.mp4"
    )
    model_ensemble_qp.animate_evolution_from_history(
        qp_state_history, "./out/model_evolution_qp.mp4"
    )
    model_truth.animate_evolution(False)


if __name__ == "__main__":
    app()
