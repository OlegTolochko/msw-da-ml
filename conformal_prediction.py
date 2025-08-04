import numpy as np
from sklearn.model_selection import train_test_split
from typer import Typer
from settings import load_settings

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

    quantiles = np.tile(
        np.expand_dims(quantiles, (0, -2, -1)),
        (cnn_test.shape[0], 1, 1, cnn_test.shape[-2], cnn_test.shape[-1]),
    )
    upper_quantiles = cnn_test + quantiles
    lower_quantiles = cnn_test - quantiles

    coverage = check_coverage(qpens_test, upper_quantiles, lower_quantiles)
    print(
        f"Target coverage: {config.calibration_quantile:.0%}, Actual coverage: {coverage:.2%}"
    )


def check_coverage(
    test_set: np.ndarray, upper_quantiles: np.ndarray, lower_quantiles: np.ndarray
):
    inside_quantile_range = (test_set > lower_quantiles) & (test_set < upper_quantiles)

    coverage = np.mean(inside_quantile_range)
    return coverage


def calibrate(truth_calib: np.ndarray, cnn_calib: np.ndarray):
    """
    Calculates calibration quantiles for conformal prediction.

    Returns:
        np.ndarray: Quantiles for each variable. Shape: (num_timesteps, 3)
    """
    # caluclate means for ensemble dim (-1)
    truth_ens_mean = np.mean(truth_calib, axis=(-1))
    cnn_ens_mean = np.mean(cnn_calib, axis=(-1))

    cnn_truth_mean_diff = np.abs(truth_calib - cnn_calib)
    # wind, water height and rain quantiles for each timestep
    quantiles = np.quantile(
        cnn_truth_mean_diff, q=config.calibration_quantile, axis=(0, -2, -1)
    )  # shape: (num_timesteps, 3)

    return quantiles


if __name__ == "__main__":
    app()
