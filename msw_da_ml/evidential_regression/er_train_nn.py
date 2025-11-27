import os
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split
import numpy as np
from cyclopts import App
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from msw_da_ml.msw.msw_data_generation import DataGenerationPipeline, DataGenerationState
from msw_da_ml.settings import load_settings, get_output_dir
from msw_da_ml.evidential_regression.mcdo_network import MCDOCNNModel
from msw_da_ml.evidential_regression.losses import GaussianNLL


app = App()

settings = load_settings()
training_config = settings.training_config

trained_nn_model_path = get_output_dir(settings.global_config.trained_nn_model_out_filename)
normalization_path = get_output_dir(settings.global_config.normalization_out_filename)


@app.command()
def load_generated_data(pipeline_state_name: str):
    data = DataGenerationPipeline.load_pipeline_state(
        pipeline_state_name=pipeline_state_name
    )
    print(data.observation_locations.shape)
    return data


@app.command()
def get_train_val_loaders(
    pipeline_state_name: str, device: str, model_name: str, normalization_path: str
):
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

    kf_train_tensor = torch.tensor(
        kf_train, dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)
    qp_train_tensor = torch.tensor(
        qp_train, dtype=torch.float32, device=device
    ).permute(0, 3, 1, 2)
    kf_val_tensor = torch.tensor(kf_val, dtype=torch.float32, device=device).permute(
        0, 3, 1, 2
    )
    qp_val_tensor = torch.tensor(qp_val, dtype=torch.float32, device=device).permute(
        0, 3, 1, 2
    )

    # Flatten ensemble dimension with batch dimension: (Batch, Channels, Length)
    kf_train_flat = kf_train_tensor.flatten(0, 1)
    qp_train_flat = qp_train_tensor.flatten(0, 1)
    kf_val_flat = kf_val_tensor.flatten(0, 1)
    qp_val_flat = qp_val_tensor.flatten(0, 1)

    # cacluate means and standard deviations of utilized dataset
    mean_in = torch.mean(kf_train_flat, dim=(0, 2), keepdim=True)
    std_in = torch.std(kf_train_flat, dim=(0, 2), keepdim=True)

    mean_out = torch.mean(qp_train_flat, dim=(0, 2), keepdim=True)
    std_out = torch.std(qp_train_flat, dim=(0, 2), keepdim=True)

    eps = 1e-8
    std_in[std_in < eps] = 1.0
    std_out[std_out < eps] = 1.0

    stats_path = os.path.join(normalization_path, f"norm_{model_name}.pt")
    torch.save(
        {
            "mean_in": mean_in,
            "std_in": std_in,
            "mean_out": mean_out,
            "std_out": std_out,
        },
        stats_path,
    )
    print(f"Saved normalization stats to {stats_path}")

    # normalization
    kf_train_norm = (kf_train_flat - mean_in) / (std_in + eps)
    kf_val_norm = (kf_val_flat - mean_in) / (std_in + eps)
    qp_train_norm = (qp_train_flat - mean_out) / (std_out + eps)
    qp_val_norm = (qp_val_flat - mean_out) / (std_out + eps)

    train_dataset = TensorDataset(kf_train_norm, qp_train_norm)
    val_dataset = TensorDataset(kf_val_norm, qp_val_norm)

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
def train_mcdo_nn(generated_training_data_name: str, model_name: str = "mcdo_cnn_model", include_timestamp_in_name: bool = True):
    """
    Trains the MCDO CNN Model based on training data given from a pipeline state.
    Saves the trained model weights under the trained_nn_model_path set in the config.
    """
    dropout = training_config.mcdo_dropout
    model = MCDOCNNModel(dropout=dropout)
    train(generated_training_data_name, model, model_name, include_timestamp_in_name)


def train(generated_training_data_name: str, model: torch.nn.Module, model_name: str, include_timestamp_in_name: bool):
    """
    Base Training method
    """
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if include_timestamp_in_name:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        model_name += f"_{timestamp}"
 
    model = model.to(device)
    train_lodar, val_loader = get_train_val_loaders(
        generated_training_data_name, device, model_name, normalization_path=normalization_path
    )

    criterion = GaussianNLL()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=training_config.learning_rate
    )

    process_bar = tqdm(range(training_config.epochs), desc="Training CNN Model")
    for epoch in process_bar:
        summed_train_loss = 0
        num_processed_train = 0

        # main training loop
        model.train()
        for kf_train_batch, qp_train_batch in train_lodar:
            model.zero_grad()

            pred_mean, pred_var = model(kf_train_batch)
            loss = criterion(qp_train_batch, pred_mean, pred_var)
            summed_train_loss += loss
            num_processed_train += 1
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        summed_val_loss = 0
        num_processed_val = 0

        # calculate loss on validation data
        model.eval()
        with torch.no_grad():
            for kf_val_batch, qp_val_batch in val_loader:
                pred_mean_val, pred_var_val = model(kf_val_batch)
                loss = criterion(qp_val_batch, pred_mean_val, pred_var_val)
                summed_val_loss += loss
                num_processed_val += 1

        avg_loss_train = summed_train_loss / num_processed_train
        avg_loss_val = summed_val_loss / num_processed_val
        process_bar.set_postfix(
            {"Train Loss": f"{avg_loss_train:.4f}", "Val Loss": f"{avg_loss_val:.4f}"}
        )

    model_name += ".pth"
    model_save_path = os.path.join(trained_nn_model_path, model_name)

    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


if __name__ == "__main__":
    app()
