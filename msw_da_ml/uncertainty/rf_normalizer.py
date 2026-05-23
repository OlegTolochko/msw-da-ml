import os
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from cyclopts import App

from msw_da_ml.data.evaluation_sequences import load_evaluation_sequences
from msw_da_ml.data.training_sequences import TrainingSequenceGenerator
from msw_da_ml.inference.cnn_sequence import load_trained_model
from msw_da_ml.settings import load_settings, get_output_dir

app = App()
settings = load_settings()
global_config = settings.global_config


def get_rf_model_path():
    model_dir = get_output_dir("models")
    return os.path.join(model_dir, "rf_error_model.joblib")


@app.command()
def train_rf(training_sequence_name: str, cnn_model_name: str):
    """
    Trains a Random Forest to predict absolute residuals (errors).
    """
    print(f"Loading training sequence from {training_sequence_name}")
    data = TrainingSequenceGenerator.load_training_sequence(training_sequence_name)

    kf_data = np.array(data.histories["kf"])
    qp_data = np.array(data.histories["qp"])
    obs_locs = np.array(data.histories["observation_locations"])

    num_ensemble_members = kf_data.shape[3]

    # obs_locs: (T, 3, G)
    # kf_data: (T, V, G, E)
    obs_locs_tiled = np.tile(
        np.expand_dims(obs_locs[:, 2:3, :], axis=-1),
        (1, 1, 1, num_ensemble_members),
    )
    # kf_with_obs: (T, V+1, G, E)
    kf_with_obs = np.concatenate([kf_data, obs_locs_tiled], axis=1)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, norm_stats = load_trained_model(
        load_model_name=cnn_model_name, device=device
    )
    model.eval()

    # flattened for model input (T*E, V+1, G)
    T, V_plus_1, G, E = kf_with_obs.shape
    kf_flat = kf_with_obs.transpose(0, 3, 1, 2).reshape(-1, V_plus_1, G)

    kf_tensor = torch.tensor(kf_flat, dtype=torch.float32, device=device)
    kf_norm = (kf_tensor - norm_stats["mean_in"]) / (norm_stats["std_in"] + 1e-8)

    print("Running CNN inference on training data")
    with torch.no_grad():
        cnn_preds_norm = model(kf_norm)

    cnn_preds = (cnn_preds_norm * norm_stats["std_out"]) + norm_stats["mean_out"]
    cnn_preds = cnn_preds.cpu().numpy()  # (T*E, V, G)

    # qp_flat: (T*E, V, G)
    qp_flat = qp_data.transpose(0, 3, 1, 2).reshape(-1, qp_data.shape[1], G)

    residuals = np.abs(qp_flat - cnn_preds)

    # X: orig. input data (T*E*G, V) w.o. obs locations
    # Y: abs residuals (T*E*G, V)
    X = (
        kf_flat[:, : qp_data.shape[1], :]
        .transpose(0, 2, 1)
        .reshape(-1, qp_data.shape[1])
    )
    Y = residuals.transpose(0, 2, 1).reshape(-1, residuals.shape[1])

    print(f"Training RF on {X.shape[0]} samples")

    rf = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, n_jobs=-1, verbose=1
    )
    rf.fit(X, Y)

    save_path = get_rf_model_path()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf, save_path)
    print(f"Random Forest model saved to {save_path}")


@app.command()
def train_rf_from_sequence(sequence_name: str):
    """
    Trains a Random Forest to predict absolute residuals from evaluation sequences.
    """
    print(f"Loading evaluation sequences from {sequence_name}")
    histories = load_evaluation_sequences(sequence_name)

    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_hist = np.asarray([history.cnn_analysis for history in histories])

    residuals = np.abs(qpens_hist - cnn_hist)

    X = cnn_hist.transpose(0, 1, 3, 4, 2).reshape(-1, 3)
    Y = residuals.transpose(0, 1, 3, 4, 2).reshape(-1, 3)

    print(f"Training RF on {X.shape[0]} samples")
    rf = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, n_jobs=-1, verbose=1
    )
    rf.fit(X, Y)

    save_path = get_rf_model_path()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf, save_path)
    print(f"Random Forest model saved to {save_path}")


if __name__ == "__main__":
    app()
