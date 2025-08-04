import numpy as np
from sklearn.model_selection import train_test_split
from typer import Typer

from test_pipeline import load_histories, ExperimentHistory
from settings import load_settings

app = Typer()

settings = load_settings()
config = settings.conformal_prediction_config


@app.command()
def conformal_prediction(hist_name: str):
    histories = load_histories(hist_name)
    truth_hist = np.asarray([history.truth for history in histories])
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_hist = np.asarray([history.cnn_analysis for history in histories])

    truth_calib, truth_test, qpens_calib, qpens_test, cnn_calib, cnn_test = (
        train_test_split(
            truth_hist,
            qpens_hist,
            cnn_hist,
            test_size=1 - config.calibration_split_ratio,
            random_state=config.calibration_split_seed,
        )
    )

    quantiles = calibrate(truth_calib=qpens_calib, cnn_calib=cnn_calib)


def calibrate(truth_calib: np.ndarray, cnn_calib: np.ndarray):
    """
    Calculates calibration quantiles for conformal prediction.

    Returns:
        np.ndarray: Quantiles for each variable. Shape: (num_timesteps, 3)
    """
    # caluclate means for seed dim (0) and ensemble dim (-1)
    truth_seedwise_mean = np.mean(truth_calib, axis=(0, -1))
    cnn_seedwise_mean = np.mean(cnn_calib, axis=(0, -1))

    cnn_truth_mean_diff = np.abs(truth_seedwise_mean - cnn_seedwise_mean)
    # wind, water height and rain quantiles for each timestep
    quantiles = np.quantile(
        cnn_truth_mean_diff, q=config.calibration_quantile, axis=-1
    )  # shape: (num_timesteps, 3)

    return quantiles


if __name__ == "__main__":
    app()
