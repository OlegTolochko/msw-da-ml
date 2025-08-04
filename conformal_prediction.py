from types import List

import numpy as np
from sklearn.model_selection import train_test_split

from test_pipeline import load_histories, ExperimentHistory
from settings import load_settings

settings = load_settings()
config = settings.conformal_prediction_config


def conformal_prediction():
    histories = load_histories()
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

    u_quantile, h_quantile, r_quantile = calibrate(
        truth_calib=qpens_calib, cnn_calib=cnn_calib
    )


def calibrate(truth_calib: np.ndarray, cnn_calib: np.ndarray):
    truth_seedwise_mean = np.mean(truth_calib, axis=0)
    cnn_seedwise_mean = np.mean(cnn_calib, axis=0)

    cnn_truth_mean_diff = np.abs(truth_seedwise_mean - cnn_seedwise_mean)
    u_quantile, h_quantile, r_quantile = np.quantile(
        cnn_truth_mean_diff, q=config.calibration_quantile, axis=0
    )

    return u_quantile, h_quantile, r_quantile
