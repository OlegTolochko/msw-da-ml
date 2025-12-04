import os
import copy

import torch
import numpy as np
from cyclopts import App

from msw_da_ml.conformal_quantile_regression.qr_network import QuantileCNNModel
from msw_da_ml.settings import load_settings, get_output_dir

app = App()

settings = load_settings()
inference_config = settings.inference_config

trained_quantile_nn_model_path = get_output_dir(
    settings.global_config.trained_quantile_nn_model_out_filename
)
quantile_normalization_path = get_output_dir(
    settings.global_config.quantile_normalization_out_filename
)


def get_most_recent_cqr_model_name():
    most_recent_model = None
    most_recent_time = 0
    for model in os.scandir(trained_quantile_nn_model_path):
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

    model_load_path = os.path.join(trained_quantile_nn_model_path, load_model_name)

    model = QuantileCNNModel()
    state_dict = torch.load(model_load_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(device)
    model.eval()

    load_normalization_name = load_model_name.removesuffix(".pth")
    norm_stats_path = os.path.join(
        quantile_normalization_path, f"norm_{load_normalization_name}.pt"
    )
    norm_stats = torch.load(norm_stats_path, map_location=device)

    return model, norm_stats
