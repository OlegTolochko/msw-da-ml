import os
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split
import numpy as np
from cyclopts import App
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from msw_da_ml.data.training_sequences import TrainingSequenceGenerator
from msw_da_ml.settings import load_settings, get_model_artifact_paths
from msw_da_ml.models.cnn import CNNModel
from msw_da_ml.training.losses import RMSEBiasLoss


app = App()

settings = load_settings()
training_config = settings.training_config

@app.command()
def load_generated_data(training_sequence_name: str):
    data = TrainingSequenceGenerator.load_training_sequence(training_sequence_name)
    print(data.observation_locations.shape)
    return data


@app.command()
def get_train_val_loaders(
    training_sequence_name: str,
    device: str,
    model_name: str,
    normalization_path: str,
    validation_sequence_name: str = "",
    skip_initial_cycles: int | None = None,
):
    """
    Returns train_loader and val_loader with Tensors of shape:
        (batch_size, num_tracked_variables, num_grid_cells)
    """
    if skip_initial_cycles is None:
        skip_initial_cycles = training_config.spinup_cycles

    train_data = TrainingSequenceGenerator.load_training_sequence(
        training_sequence_name
    )
    kf_train_all, qp_train_all = _sequence_to_supervised_arrays(
        train_data, skip_initial_cycles=skip_initial_cycles
    )

    if validation_sequence_name:
        validation_data = TrainingSequenceGenerator.load_training_sequence(
            validation_sequence_name
        )
        kf_train, qp_train = kf_train_all, qp_train_all
        kf_val, qp_val = _sequence_to_supervised_arrays(
            validation_data, skip_initial_cycles=skip_initial_cycles
        )
    else:
        kf_train, kf_val, qp_train, qp_val = train_test_split(
            kf_train_all,
            qp_train_all,
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

    eps = 1e-8
    mean_in, std_in, mean_out, std_out = _paper_normalization_stats(kf_train_flat, eps)
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
        f"Initialized train-val loaders, with: {len(kf_train_flat)} samples "
        + f"in the training set and {len(kf_val_flat)} samples in the validation set"
    )
    return train_loader, val_loader


def _sequence_to_supervised_arrays(data, skip_initial_cycles: int):
    kf_data = np.array(data.histories["kf"])[skip_initial_cycles:]
    qp_data = np.array(data.histories["qp"])[skip_initial_cycles:]
    observation_locations = np.array(data.histories["observation_locations"])[
        skip_initial_cycles:
    ]
    if len(kf_data) == 0:
        raise ValueError(
            "No training cycles remain after applying "
            f"skip_initial_cycles={skip_initial_cycles}."
        )

    num_ensemble_members = kf_data[0].shape[2]
    # The authors' CNN input uses obspos[2*nx:], where 0 means rainy/observed
    # and 1 means dry/unobserved. Internally this project stores True=observed.
    rain_unobserved_indicator = np.logical_not(observation_locations[:, 2:3])
    observation_locations_data = np.tile(
        np.expand_dims(rain_unobserved_indicator, axis=-1),
        (1, 1, 1, num_ensemble_members),
    )
    kf_data_with_observation_locations = np.concat(
        [kf_data, observation_locations_data], axis=1
    )
    return kf_data_with_observation_locations, qp_data


def _paper_normalization_stats(kf_train_flat: torch.Tensor, eps: float):
    state_train = kf_train_flat[:, :3]
    state_mean = torch.mean(state_train, dim=(0, 2), keepdim=True)
    state_mean[:, 2] = 0.0

    # Match the authors' np.average(np.var(X[..., :3], axis=0), axis=0).
    state_var_by_grid = torch.var(state_train, dim=0, unbiased=False)
    state_std = torch.sqrt(torch.mean(state_var_by_grid, dim=1)).view(1, 3, 1)
    state_std[state_std < eps] = 1.0

    obs_mean = torch.zeros(
        (1, 1, 1), dtype=kf_train_flat.dtype, device=kf_train_flat.device
    )
    obs_std = torch.ones(
        (1, 1, 1), dtype=kf_train_flat.dtype, device=kf_train_flat.device
    )
    mean_in = torch.cat([state_mean, obs_mean], dim=1)
    std_in = torch.cat([state_std, obs_std], dim=1)

    return mean_in, std_in, state_mean.clone(), state_std.clone()


@app.command()
def train_nn(
    training_sequence_name: str,
    model_name: str = "cnn_model",
    validation_sequence_name: str = "",
    bias_loss_weight: float | None = None,
    skip_initial_cycles: int | None = None,
    include_timestamp_in_name: bool = True,
):
    """
    Trains the CNN Model based on a generated training sequence.
    Saves the trained model weights under the trained_nn_model_path set in the config.
    """
    model = CNNModel()
    train(
        training_sequence_name,
        model,
        model_name,
        include_timestamp_in_name,
        validation_sequence_name=validation_sequence_name,
        bias_loss_weight=bias_loss_weight,
        skip_initial_cycles=skip_initial_cycles,
    )


def train(
    training_sequence_name: str,
    model: torch.nn.Module,
    model_name: str,
    include_timestamp_in_name: bool,
    validation_sequence_name: str = "",
    bias_loss_weight: float | None = None,
    skip_initial_cycles: int | None = None,
):
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

    model_save_path, norm_stats_path = get_model_artifact_paths(model_name)

    model = model.to(device)
    train_lodar, val_loader = get_train_val_loaders(
        training_sequence_name,
        device,
        model_name,
        normalization_path=model_save_path.parent,
        validation_sequence_name=validation_sequence_name,
        skip_initial_cycles=skip_initial_cycles,
    )

    criterion = RMSEBiasLoss(bias_loss_weight=bias_loss_weight)
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

            pred_states_train = model(kf_train_batch)
            loss = criterion(qp_train_batch, pred_states_train)
            summed_train_loss += loss.detach()
            num_processed_train += 1
            loss.backward()
            optimizer.step()

        summed_val_loss = 0
        num_processed_val = 0

        # calculate loss on validation data
        model.eval()
        with torch.no_grad():
            for kf_val_batch, qp_val_batch in val_loader:
                pred_state_val = model(kf_val_batch)
                loss = criterion(qp_val_batch, pred_state_val)
                summed_val_loss += loss.detach()
                num_processed_val += 1

        avg_loss_train = summed_train_loss / num_processed_train
        avg_loss_val = summed_val_loss / num_processed_val
        process_bar.set_postfix(
            {"Train Loss": f"{avg_loss_train:.4f}", "Val Loss": f"{avg_loss_val:.4f}"}
        )

    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")
    print(f"Saved matching normalization stats to {norm_stats_path}.")


if __name__ == "__main__":
    app()
