import os
import sys
import copy

import torch
import numpy as np
from typer import Typer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantile_regression.quantile_network import QuantileCNNModel
from core.settings import load_settings

app = Typer()

settings = load_settings()
inference_config = settings.inference_config

trained_quantile_nn_model_out_filename = (
    settings.global_config.trained_quantile_nn_model_out_filename
)
trained_quantile_nn_model_path = (
    f"{settings.global_config.out_path}{trained_quantile_nn_model_out_filename}"
)

quantile_normalization_out_filename = (
    settings.global_config.quantile_normalization_out_filename
)
quantile_normalization_path = (
    f"{settings.global_config.out_path}{quantile_normalization_out_filename}"
)


def get_most_recent_model_name():
    most_recent_model = None
    most_recent_time = 0
    for model in os.scandir(trained_quantile_nn_model_path):
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

    model_load_path = f"{trained_quantile_nn_model_path}{load_model_name}"

    model = QuantileCNNModel()
    state_dict = torch.load(model_load_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(device)
    model.eval()

    return model, load_model_name


def load_normalization(load_model_name: str, device):
    load_model_name = load_model_name.removesuffix(".pth")
    stats_path = os.path.join(quantile_normalization_path, f"{load_model_name}.pt")
    norm_stats = torch.load(stats_path, map_location=device)
    return norm_stats


def load_trained_model(load_model_name: str, device):
    if not load_model_name:
        load_model_name = get_most_recent_model_name()
        if not load_model_name:
            raise FileNotFoundError("No model files found")

    if not load_model_name.endswith(".pth"):
        load_model_name += ".pth"

    model_load_path = f"{trained_quantile_nn_model_path}{load_model_name}"

    model = QuantileCNNModel()
    state_dict = torch.load(model_load_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(device)
    model.eval()

    return model, load_model_name
