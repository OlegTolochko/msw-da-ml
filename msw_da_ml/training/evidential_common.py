from datetime import datetime

import torch
from cyclopts import App
from tqdm import tqdm

from msw_da_ml.data.training_sequences import TrainingSequenceGenerator
from msw_da_ml.settings import load_settings, get_model_artifact_paths
from msw_da_ml.models.mcdo import MCDOCNNModel
from msw_da_ml.models.evidential import NIGCNNModel
from msw_da_ml.training.train_cnn import get_train_val_loaders
from msw_da_ml.training.evidential_losses import GaussianNLL, NIGLoss


app = App()

settings = load_settings()
training_config = settings.training_config

@app.command()
def load_generated_data(training_sequence_name: str):
    data = TrainingSequenceGenerator.load_training_sequence(training_sequence_name)
    print(data.observation_locations.shape)
    return data


@app.command()
def train_mcdo_nn(
    training_sequence_name: str,
    model_name: str = "mcdo_cnn_model",
    validation_sequence_name: str = "",
    skip_initial_cycles: int | None = None,
    include_timestamp_in_name: bool = True,
):
    """
    Trains the MCDO CNN Model based on a generated training sequence.
    Saves the trained model weights under the trained_nn_model_path set in the config.
    """
    dropout = training_config.mcdo_dropout
    model = MCDOCNNModel(dropout=dropout)
    train(
        training_sequence_name,
        model,
        model_name,
        include_timestamp_in_name,
        warmup=True,
        validation_sequence_name=validation_sequence_name,
        skip_initial_cycles=skip_initial_cycles,
    )


@app.command()
def train_evidential_nn(
    training_sequence_name: str,
    model_name: str = "evidential_cnn_model",
    reg_coef: float = 1.0,
    validation_sequence_name: str = "",
    skip_initial_cycles: int | None = None,
    include_timestamp_in_name: bool = True,
):
    """
    Trains the NIG CNN Model based on a generated training sequence.
    Saves the trained model weights under the trained_nn_model_path set in the config.
    """
    model = NIGCNNModel()
    train(
        training_sequence_name,
        model,
        model_name,
        include_timestamp_in_name,
        warmup=False,
        nig_reg_coef=reg_coef,
        validation_sequence_name=validation_sequence_name,
        skip_initial_cycles=skip_initial_cycles,
    )


def train(
    training_sequence_name: str,
    model: torch.nn.Module,
    model_name: str,
    include_timestamp_in_name: bool,
    warmup: bool,
    nig_reg_coef: float = 1.0,
    validation_sequence_name: str = "",
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

    nll_criterion = GaussianNLL()
    nig_criterion = NIGLoss(reg_coef=nig_reg_coef)
    if warmup:
        mse_criterion = torch.nn.MSELoss()
        warmup_epochs = 20
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

            if isinstance(model, NIGCNNModel):
                pred_gamma, pred_nu, pred_alpha, pred_beta = model(kf_train_batch)
                loss = nig_criterion(
                    pred_gamma, pred_nu, pred_alpha, pred_beta, qp_train_batch
                )
            else:
                pred_mean, pred_logvar = model(kf_train_batch)
                if warmup:
                    if epoch < warmup_epochs:
                        loss = mse_criterion(pred_mean, qp_train_batch)
                    else:
                        loss = nll_criterion(qp_train_batch, pred_mean, pred_logvar)
                else:
                    loss = nll_criterion(qp_train_batch, pred_mean, pred_logvar)

            summed_train_loss += loss.detach()
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
                if isinstance(model, NIGCNNModel):
                    pred_gamma_val, pred_nu_val, pred_alpha_val, pred_beta_val = model(
                        kf_val_batch
                    )
                    loss = nig_criterion(
                        pred_gamma_val,
                        pred_nu_val,
                        pred_alpha_val,
                        pred_beta_val,
                        qp_val_batch,
                    )
                else:
                    pred_mean_val, pred_logvar_val = model(kf_val_batch)
                    if warmup:
                        if epoch < warmup_epochs:
                            loss = mse_criterion(pred_mean_val, qp_val_batch)
                        else:
                            loss = nll_criterion(
                                qp_val_batch, pred_mean_val, pred_logvar_val
                            )
                    else:
                        loss = nll_criterion(
                            qp_val_batch, pred_mean_val, pred_logvar_val
                        )
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
