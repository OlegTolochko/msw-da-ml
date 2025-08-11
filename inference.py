import os
import copy

import torch
import numpy as np
from typer import Typer

from models.network import CNNModel
from core.settings import load_settings
from assimilation.assimilation import EnsembleKalmanFilter, QPEnsemble
from data.observation_generation import ObservationGenerator
from models.msw_model import EnsembleModel
from core.random_manager import RandomGenerators
from analysis.msw_model_visualization import ModelComparatorVisualizer
from data.msw_data_generation import DataGenerationPipeline

app = Typer()

settings = load_settings()
inference_config = settings.inference_config

trained_nn_model_out_filename = settings.global_config.trained_nn_model_out_filename
trained_nn_model_path = (
    f"{settings.global_config.out_path}{trained_nn_model_out_filename}"
)

normalization_out_filename = settings.global_config.normalization_out_filename
normalization_path = f"{settings.global_config.out_path}{normalization_out_filename}"


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

    return model, load_model_name


def load_normalization(load_model_name: str, device):
    load_model_name = load_model_name.removesuffix(".pth")
    stats_path = os.path.join(normalization_path, f"{load_model_name}.pt")
    norm_stats = torch.load(stats_path, map_location=device)
    return norm_stats


@app.command()
def inference(
    num_inference_steps: int, load_model_name: str = "", compute_qpens: bool = True
):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, load_model_name = load_trained_model(load_model_name, device)

    norm_stats = load_normalization(load_model_name, device)
    mean_in = norm_stats["mean_in"]
    std_in = norm_stats["std_in"]
    mean_out = norm_stats["mean_out"]
    std_out = norm_stats["std_out"]

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

    if compute_qpens:
        qpens = QPEnsemble()
        qpens_model = copy.deepcopy(ensemble_model)

    for i in range(num_inference_steps):
        truth_model.propagate()
        ensemble_model.propagate()

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
        assimilated_tensor_norm = (assimilated_tensor - mean_in) / (std_in + 1e-8)

        with torch.no_grad():
            corrected_tensor_norm = model(assimilated_tensor_norm)

        corrected_tensor = (corrected_tensor_norm * std_out) + mean_out

        corrected_state = corrected_tensor.squeeze(1).permute(1, 2, 0).cpu().numpy()

        if compute_qpens:
            qpens_model.propagate()
            qpens_state = qpens_model.get_state()
            assimilated_qpens_state = qpens.assimilate(
                qpens_state, obs_data.observation, obs_data.locations
            )
            qpens_model.assimilate(assimilated_qpens_state)

        ensemble_model.assimilate(corrected_state)

    visualize_update_performance(ensemble_model, truth_model, load_model_name)
    if compute_qpens:
        visualize_update_performance(
            ensemble_model, qpens_model, load_model_name, "CNN", "QPEns"
        )


def compare_models():
    pass


@app.command()
def visualize_from_model(model_name: str = "pipeline_state.pkl"):
    data = DataGenerationPipeline.load_pipeline_state(pipeline_state_name=model_name)

    truth_model = data.models["truth"]
    qp_model = data.models["ensemble_qp"]
    visualize_update_performance(qp_model, truth_model, model_name)


def visualize_update_performance(
    cnn_model: EnsembleModel,
    truth_model: EnsembleModel,
    trained_model_name: str,
    model_name1: str = "CNN",
    model_name2: str = "Truth",
):
    cnn_history = cnn_model.get_history()
    truth_history = truth_model.get_history()
    print(len(cnn_history))
    print(len(truth_history))
    min_len = min(len(cnn_history), len(truth_history))

    visualizer = ModelComparatorVisualizer(
        history1=cnn_history[:min_len],
        history2=truth_history[:min_len],
        model1_name=model_name1,
        model2_name=model_name2,
    )
    visualizer.animate(save_name=trained_model_name)


if __name__ == "__main__":
    app()
