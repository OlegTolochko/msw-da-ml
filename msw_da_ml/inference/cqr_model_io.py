import os

import torch
from cyclopts import App

from msw_da_ml.models.cqr import QuantileCNNModel
from msw_da_ml.settings import get_model_artifact_dir, get_model_artifact_paths, load_settings

app = App()

settings = load_settings()
inference_config = settings.inference_config

def get_most_recent_cqr_model_name():
    most_recent_model = None
    most_recent_time = 0
    for model in os.scandir(get_model_artifact_dir("cqr")):
        if model.is_file() and model.name.endswith(".pth"):
            mod_time = model.stat().st_mtime_ns
            if mod_time > most_recent_time:
                most_recent_model = model
                most_recent_time = mod_time
    return os.path.basename(most_recent_model.path) if most_recent_model else None


def load_cqr_trained_model(load_model_name: str, device):
    if not load_model_name:
        load_model_name = get_most_recent_cqr_model_name()
        if not load_model_name:
            raise FileNotFoundError("No model files found")

    if not load_model_name.endswith(".pth"):
        load_model_name += ".pth"

    model_load_path, norm_stats_path = get_model_artifact_paths(load_model_name)

    model = QuantileCNNModel()
    state_dict = torch.load(model_load_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(device)
    model.eval()

    norm_stats = torch.load(norm_stats_path, map_location=device)

    return model, norm_stats
