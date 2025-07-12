import torch
from sklearn.model_selection import train_test_split
import numpy as np
from data_generation_pipeline import DataGenerationPipeline, DataGenerationState
from typer import Typer

from settings import load_settings

from torch.utils.data import DataLoader, TensorDataset

app = Typer()

settings = load_settings()
training_config = settings.training_config


@app.command()
def load_generated_data(pipeline_state_name: str):
    data = DataGenerationPipeline.load_pipeline_state(
        pipeline_state_name=pipeline_state_name
    )
    print(data.observation_locations.shape)
    return data


@app.command()
def get_train_val_loaders(pipeline_state_name: str):
    """
    Returns train_loader and val_loader with Tensors of shape:
        (batch_size, num_tracked_variables, num_grid_cells)
    """
    data = DataGenerationPipeline.load_pipeline_state(
        pipeline_state_name=pipeline_state_name
    )

    kf_data = np.array(data.histories["kf"])
    qp_data = np.array(data.histories["qp"])

    num_ensemble_members = kf_data[0].shape[2]
    observation_locations_data = np.array(data.histories["observation_locations"])
    # reshape observation locations data to only have rain locations and same shape as kf_data
    observation_locations_data = np.tile(
        np.expand_dims(observation_locations_data[:, 2:3], axis=-1),
        (1, 1, 1, num_ensemble_members),
    )
    kf_data_with_observation_locations = np.concat(
        [kf_data, observation_locations_data], axis=1
    )

    kf_train, kf_val, qp_train, qp_val = train_test_split(
        kf_data_with_observation_locations,
        qp_data,
        test_size=training_config.val_split_size,
        random_state=training_config.random_state_train_test_split,
    )

    kf_train_tensor = torch.tensor(kf_train, dtype=torch.float32).permute(0, 3, 1, 2)
    qp_train_tensor = torch.tensor(qp_train, dtype=torch.float32).permute(0, 3, 1, 2)
    kf_val_tensor = torch.tensor(kf_val, dtype=torch.float32).permute(0, 3, 1, 2)
    qp_val_tensor = torch.tensor(qp_val, dtype=torch.float32).permute(0, 3, 1, 2)

    kf_train_flat = kf_train_tensor.flatten(0, 1)
    qp_train_flat = qp_train_tensor.flatten(0, 1)
    kf_val_flat = kf_val_tensor.flatten(0, 1)
    qp_val_flat = qp_val_tensor.flatten(0, 1)

    train_dataset = TensorDataset(kf_train_flat, qp_train_flat)
    val_dataset = TensorDataset(kf_val_flat, qp_val_flat)

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=False
    )

    print(
        f"Initialized train-val loaders, with: {len(kf_train)} pairs "
        + f"in the training set and {len(kf_val)} pairs in the validation set"
    )
    return train_loader, val_loader


@app.command()
def train_nn():
    train_lodar, val_loader = get_train_val_loaders()
    pass


if __name__ == "__main__":
    app()
