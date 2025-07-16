import os

import torch

from network import CNNModel
from settings import load_settings
from assimilation import kf_assimilate
from msw_model import ModifiedShallowWaterModel

settings = load_settings()
training_config = settings.training_config

trained_nn_model_out_filename = settings.global_config.trained_nn_model_out_filename
trained_nn_model_path = (
    f"{settings.global_config.out_path}{trained_nn_model_out_filename}"
)


def get_most_recent_model_name():
    most_recent_model = None
    most_recent_time = 0
    for model in os.scandir(trained_nn_model_path):
        if model.is_file():
            mod_time = model.stat().st_mtime_ns
            if mod_time > most_recent_time:
                most_recent_model = model
                most_recent_time = mod_time
    return most_recent_model.path


def load_trained_model(load_model_name: str, device: any):
    if not load_model_name:
        load_model_name = get_most_recent_model_name()

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


def inference(load_model_name: str = ""):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    load_trained_model(load_model_name, device)

    truth_state = ModifiedShallowWaterModel(num_ensemble_members=1).initialize()
    ensemble_state = ModifiedShallowWaterModel(num_ensemble_members=10).initialize()


def compare_models():
    pass


def visualize_update_performance():
    pass
