import os
from datetime import datetime

import torch
from typer import Typer
from tqdm import tqdm

from msw_da_ml.settings import load_settings
from msw_da_ml.conformal_quantile_regression.qr_network import QuantileCNNModel
from msw_da_ml.conformal_quantile_regression.cqr_losses import pinball_loss
from msw_da_ml.msw_cnn.train_nn import get_train_val_loaders


app = Typer()

settings = load_settings()
training_config = settings.training_config

trained_quantile_nn_model_out_filename = (
    settings.global_config.trained_quantile_nn_model_out_filename
)
trained_quantile_nn_model_path = (
    f"{settings.global_config.out_path}{trained_quantile_nn_model_out_filename}"
)
os.makedirs(trained_quantile_nn_model_path, exist_ok=True)

quantile_normalization_out_filename = (
    settings.global_config.quantile_normalization_out_filename
)
quantile_normalization_path = (
    f"{settings.global_config.out_path}{quantile_normalization_out_filename}"
)
os.makedirs(quantile_normalization_path, exist_ok=True)


@app.command()
def train_quantile_nn(
    pipeline_state_name: str,
    quantile_tau: float = 0.9,
    include_timestamp_in_name: bool = True,
):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_name = f"quantile_{training_config.model_save_name}"
    if include_timestamp_in_name:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        model_name += f"_{timestamp}"

    model = QuantileCNNModel()
    model = model.to(device)
    train_lodar, val_loader = get_train_val_loaders(
        pipeline_state_name,
        device,
        model_name,
        normalization_path=quantile_normalization_path,
    )

    lower_quantile = 0.5 * (1 - quantile_tau)
    upper_quantile = 1 - 0.5 * (1 - quantile_tau)

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

            pred_quantile_lower_train, pred_quantile_upper_train = model(kf_train_batch)
            loss = pinball_loss(
                qp_train_batch, pred_quantile_lower_train, tau=lower_quantile
            ) + pinball_loss(
                qp_train_batch, pred_quantile_upper_train, tau=upper_quantile
            )
            summed_train_loss += loss
            num_processed_train += 1
            loss.backward()
            optimizer.step()

        summed_val_loss = 0
        num_processed_val = 0

        # calculate loss on validation data
        model.eval()
        with torch.no_grad():
            for kf_val_batch, qp_val_batch in val_loader:
                pred_quantile_lower_val, pred_quantile_upper_val = model(kf_val_batch)
                loss = pinball_loss(
                    qp_val_batch, pred_quantile_lower_val, tau=lower_quantile
                ) + pinball_loss(
                    qp_val_batch, pred_quantile_upper_val, tau=upper_quantile
                )
                summed_val_loss += loss
                num_processed_val += 1

        avg_loss_train = summed_train_loss / num_processed_train
        avg_loss_val = summed_val_loss / num_processed_val
        process_bar.set_postfix(
            {"Train Loss": f"{avg_loss_train:.4f}", "Val Loss": f"{avg_loss_val:.4f}"}
        )

    model_name += ".pth"
    model_save_path = os.path.join(trained_quantile_nn_model_path, model_name)

    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


if __name__ == "__main__":
    app()
