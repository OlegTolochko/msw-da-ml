import torch
from sklearn.model_selection import train_test_split
import numpy as np
from data_generation_pipeline import DataGenerationPipeline, DataGenerationState
from typer import Typer

from settings import load_settings

from torch.utils.data import DataLoader, TensorDataset

app = Typer()

settings = load_settings()
nn_config = settings.nn_config


@app.command()
def load_generated_data(pipeline_state_name: str):
    data = DataGenerationPipeline.load_pipeline_state(
        pipeline_state_name=pipeline_state_name
    )
    return data


@app.command()
def get_train_val_loaders(pipeline_state_name: str):
    data = DataGenerationPipeline.load_pipeline_state(
        pipeline_state_name=pipeline_state_name
    )

    kf_data = data.histories["kf"]
    qp_data = data.histories["qp"]

    kf_train, kf_val, qp_train, qp_val = train_test_split(
        kf_data,
        qp_data,
        test_size=nn_config.val_split_size,
        random_state=nn_config.random_state_train_test_split,
    )

    train_dataset = TensorDataset(
        torch.tensor(kf_train, dtype=torch.float32),
        torch.tensor(qp_train, dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(kf_val, dtype=torch.float32),
        torch.tensor(qp_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=nn_config.batch_size, shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=nn_config.batch_size, shuffle=False)

    return train_loader, val_loader


@app.command()
def train_nn():
    pass


if __name__ == "__main__":
    app()
