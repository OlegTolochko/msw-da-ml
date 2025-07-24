import os

import torch
import numpy as np
from typer import Typer

from network import CNNModel
from settings import load_settings
from assimilation import EnsembleKalmanFilter
from observation_generation import ObservationGenerator
from msw_model import EnsembleModel
from random_manager import RandomGenerators

app = Typer()

settings = load_settings()
inference_config = settings.inference_config

trained_nn_model_out_filename = settings.global_config.trained_nn_model_out_filename
trained_nn_model_path = (
    f"{settings.global_config.out_path}{trained_nn_model_out_filename}"
)


def get_most_recent_model_name():
    most_recent_model = None
    most_recent_time = 0
    for model in os.scandir(trained_nn_model_path):
        if model.is_file() and model.name.endswith(".pth"):
            mod_time = model.stat().st_mtime_ns
            if mod_time > most_recent_time:
                most_recent_model = model
                most_recent_time = mod_time
    return os.path.basename(most_recent_model.path) if most_recent_model else None


def load_trained_model(load_model_name: str, device):
    if not load_model_name:
        load_model_name = get_most_recent_model_name()
        if not load_model_name:
            raise FileNotFoundError("No model files found")

    if not load_model_name.endswith(".pth"):
        load_model_name += ".pth"

    model_load_path = f"{trained_nn_model_path}{load_model_name}"

    model = CNNModel()
    state_dict = torch.load(model_load_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(device)
    model.eval()

    return model


@app.command()
def inference(num_inference_steps: int, load_model_name: str = ""):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_trained_model(load_model_name, device)

    enkf = EnsembleKalmanFilter()
    rngs = RandomGenerators.from_seed(inference_config.inference_seed)
    obs_generator = ObservationGenerator(rngs)

    truth_model = EnsembleModel(num_ensemble_members=1, random_generator=rngs.truth_rng)
    truth_model.initialize()

    ensemble_model = EnsembleModel(
        num_ensemble_members=inference_config.num_ensemble_members,
        random_generator=rngs.ensemble_rng,
    )
    ensemble_model.initialize()

    for i in range(num_inference_steps):
        truth_model.propagate()

        truth_state = truth_model.get_state()
        ensemble_state = ensemble_model.get_state()

        obs_data = obs_generator.generate_observations_with_locations(
            truth_state, inference_config.num_ensemble_members
        )

        assimilated_state = enkf.assimilate(
            ensemble_state, obs_data.observation, obs_data.locations
        )

        observation_locations_data = np.tile(
            np.expand_dims(obs_data.locations[2:3], axis=-1),
            (1, 1, inference_config.num_ensemble_members),
        )
        assimilated_state_with_observation_locations = np.concat(
            [assimilated_state, observation_locations_data], axis=0
        )
        assimilated_tensor = torch.tensor(
            assimilated_state_with_observation_locations,
            dtype=torch.float32,
            device=device,
        ).permute(2, 0, 1)

        with torch.no_grad():
            corrected_tensor = model(assimilated_tensor)

        corrected_state = corrected_tensor.squeeze(1).permute(1, 2, 0).cpu().numpy()

        ensemble_model.assimilate(corrected_state)
        ensemble_model.propagate()


def compare_models():
    pass


def visualize_update_performance():
    pass


if __name__ == "__main__":
    app()
