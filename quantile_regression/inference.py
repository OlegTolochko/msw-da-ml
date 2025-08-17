import os
import copy

import torch
import numpy as np
from typer import Typer

from quantile_network import QuantileCNNModel
from core.settings import load_settings

app = Typer()

settings = load_settings()
inference_config = settings.inference_config

trained_quantile_nn_model_out_filename = settings.global_config.trained_quantile_nn_model_out_filename
trained_quantile_nn_model_path = (
    f"{settings.global_config.out_path}{trained_quantile_nn_model_out_filename}"
)


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
