from datetime import datetime

import torch
from cyclopts import App
from tqdm import tqdm

from msw_da_ml.models.mcdo import MCDOCNNModel
from msw_da_ml.settings import get_model_artifact_paths, load_settings
from msw_da_ml.training.evidential_losses import GaussianNLL
from msw_da_ml.training.train_cnn import get_train_val_loaders


app = App()

settings = load_settings()
training_config = settings.training_config


@app.command()
def train_deep_ensemble_nn(
    training_sequence_name: str,
    model_name: str = "ensemble_cnn_model",
    ensemble_size: int | None = None,
    validation_sequence_name: str = "",
    skip_initial_cycles: int | None = None,
    include_timestamp_in_name: bool = True,
):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if include_timestamp_in_name:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        model_name += f"_{timestamp}"

    if ensemble_size is None:
        ensemble_size = training_config.deep_ensemble_size

    model_save_path, norm_stats_path = get_model_artifact_paths(model_name)
    train_loader, val_loader = get_train_val_loaders(
        training_sequence_name,
        device,
        model_name,
        normalization_path=model_save_path.parent,
        validation_sequence_name=validation_sequence_name,
        skip_initial_cycles=skip_initial_cycles,
    )

    member_model_files = []
    for member_idx in range(ensemble_size):
        torch.manual_seed(10_000 + member_idx)
        model = MCDOCNNModel(dropout=0.0).to(device)
        state_dict = _train_member(model, train_loader, val_loader, member_idx)
        member_model_path = model_save_path.with_name(
            f"{model_save_path.stem}_member{member_idx:02d}.pth"
        )
        torch.save(state_dict, member_model_path)
        member_model_files.append(member_model_path.name)
        print(f"Saved ensemble member {member_idx + 1} to {member_model_path}.")

    torch.save(
        {
            "member_model_files": member_model_files,
            "ensemble_size": ensemble_size,
            "dropout": 0.0,
        },
        model_save_path,
    )
    print(f"Saved the deep ensemble to {model_save_path}.")
    print(f"Saved matching normalization stats to {norm_stats_path}.")


def _train_member(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    member_idx: int,
) -> dict[str, torch.Tensor]:
    nll_criterion = GaussianNLL()
    mse_criterion = torch.nn.MSELoss()
    warmup_epochs = 20
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=training_config.learning_rate
    )

    process_bar = tqdm(
        range(training_config.epochs),
        desc=f"Training ensemble member {member_idx + 1}",
    )
    for epoch in process_bar:
        summed_train_loss = 0
        num_processed_train = 0

        model.train()
        for kf_train_batch, qp_train_batch in train_loader:
            model.zero_grad()
            pred_mean, pred_logvar = model(kf_train_batch)
            if epoch < warmup_epochs:
                loss = mse_criterion(pred_mean, qp_train_batch)
            else:
                loss = nll_criterion(qp_train_batch, pred_mean, pred_logvar)

            summed_train_loss += loss.detach()
            num_processed_train += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        summed_val_loss = 0
        num_processed_val = 0

        model.eval()
        with torch.no_grad():
            for kf_val_batch, qp_val_batch in val_loader:
                pred_mean_val, pred_logvar_val = model(kf_val_batch)
                if epoch < warmup_epochs:
                    loss = mse_criterion(pred_mean_val, qp_val_batch)
                else:
                    loss = nll_criterion(qp_val_batch, pred_mean_val, pred_logvar_val)
                summed_val_loss += loss.detach()
                num_processed_val += 1

        avg_loss_train = summed_train_loss / num_processed_train
        avg_loss_val = summed_val_loss / num_processed_val
        process_bar.set_postfix(
            {"Train Loss": f"{avg_loss_train:.4f}", "Val Loss": f"{avg_loss_val:.4f}"}
        )

    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


if __name__ == "__main__":
    app()
