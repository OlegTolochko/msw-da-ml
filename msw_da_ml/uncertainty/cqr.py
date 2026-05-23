import numpy as np
from sklearn.model_selection import train_test_split
from cyclopts import App
import matplotlib.pyplot as plt

from msw_da_ml.inference.cqr_sequence import (
    load_cqr_evaluation_sequences,
)
from msw_da_ml.settings import load_settings, get_output_dir

app = App()

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config

viz_dir = get_output_dir(global_config.visualizations_out_filename)


@app.command()
def raw_quantile_conformal_prediction(sequence_name: str):
    histories = load_cqr_evaluation_sequences(sequence_name)
    truth_hist = np.asarray([history.truth for history in histories])
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_lower_hist = np.asarray(
        [history.cnn_analysis_lower_quantiles for history in histories]
    )
    cnn_upper_hist = np.asarray(
        [history.cnn_analysis_upper_quantiles for history in histories]
    )

    (
        truth_calib,
        truth_test,
        qpens_calib,
        qpens_test,
        cnn_lower_calib,
        cnn_lower_test,
        cnn_upper_calib,
        cnn_upper_test,
    ) = train_test_split(
        truth_hist,
        qpens_hist,
        cnn_lower_hist,
        cnn_upper_hist,
        test_size=1 - config.calibration_split_ratio,
        random_state=config.calibration_split_seed,
    )

    cnn_lower_mean = np.mean(cnn_lower_test, axis=(-1))
    cnn_upper_mean = np.mean(cnn_upper_test, axis=(-1))

    coverage = check_quantile_coverage(qpens_test, cnn_lower_mean, cnn_upper_mean)

    print(f"Target coverage: {config.calibration_quantile:.0%}")
    print(f"Actual coverage: {np.mean(coverage):.2%}")
    save_name = sequence_name + "_raw"

    visualize_quantile_coverage(coverage, save_name)
    visualize_quantile_intervals(cnn_lower_mean, cnn_upper_mean, save_name)
    visualize_quantile_gridpoints(
        cnn_lower_mean, cnn_upper_mean, truth_test, qpens_test, save_name
    )


@app.command()
def cqr_prediction(sequence_name: str):
    """
    Runs conformal prediction pipeline for quantile regression models.
    """
    histories = load_cqr_evaluation_sequences(sequence_name)
    truth_hist = np.asarray([history.truth for history in histories])
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_lower_hist = np.asarray(
        [history.cnn_analysis_lower_quantiles for history in histories]
    )
    cnn_upper_hist = np.asarray(
        [history.cnn_analysis_upper_quantiles for history in histories]
    )

    # Split data for calibration and testing
    (
        truth_calib,
        truth_test,
        qpens_calib,
        qpens_test,
        cnn_lower_calib,
        cnn_lower_test,
        cnn_upper_calib,
        cnn_upper_test,
    ) = train_test_split(
        truth_hist,
        qpens_hist,
        cnn_lower_hist,
        cnn_upper_hist,
        test_size=1 - config.calibration_split_ratio,
        random_state=config.calibration_split_seed,
    )

    quantile_adjustment = calibrate_quantile_intervals_symmetric(
        truth_calib=qpens_calib,  # Use QPEns as ground truth for CNN
        cnn_lower_calib=cnn_lower_calib,
        cnn_upper_calib=cnn_upper_calib,
    )

    cnn_lower_test_adjusted, cnn_upper_test_adjusted = (
        apply_symmetric_quantile_adjustments(
            cnn_lower_test, cnn_upper_test, quantile_adjustment
        )
    )

    # Check coverage
    coverage = check_quantile_coverage(
        qpens_test, cnn_lower_test_adjusted, cnn_upper_test_adjusted
    )

    print(f"Target coverage: {config.calibration_quantile:.0%}")
    print(f"Actual coverage: {np.mean(coverage):.2%}")

    visualize_quantile_coverage(coverage, sequence_name)
    visualize_quantile_intervals(
        cnn_lower_test_adjusted, cnn_upper_test_adjusted, sequence_name
    )
    visualize_quantile_gridpoints(
        cnn_lower_test_adjusted,
        cnn_upper_test_adjusted,
        truth_test,
        qpens_test,
        sequence_name,
    )


def calculate_empirical_quantile(scores: np.ndarray, alpha: float):
    """
    Calculate empirical quantile from scores.

    Returns:
        Quantile values of shape (time,)
    """
    if scores.ndim == 3:
        # Shape: (seeds, time, grid)
        num_seeds, num_time, num_grid = scores.shape
        # Reshape to (seeds * grid, time)
        s = scores.transpose(0, 2, 1).reshape(num_seeds * num_grid, num_time)
    elif scores.ndim == 4:
        # Shape: (seeds, time, grid, ens)
        num_seeds, num_time, num_grid, num_ens = scores.shape
        # Reshape to (seeds * grid * ens, time)
        s = scores.transpose(0, 2, 3, 1).reshape(
            num_seeds * num_grid * num_ens, num_time
        )
    else:
        raise ValueError(f"Unexpected scores shape: {scores.shape}")

    m = s.shape[0]
    k = int(np.ceil((1.0 - alpha) * (m + 1))) - 1
    k = np.clip(k, 0, m - 1)
    s_part = np.partition(s, k, axis=0)
    return s_part[k]  # Shape: (time,)


def calibrate_quantile_intervals_symmetric(
    truth_calib, cnn_lower_calib, cnn_upper_calib, ens_mean: bool = True
):
    """
    Calibrate quantile intervals using conformal prediction.

    Returns:
        Tuple of adjustment factors for lower and upper quantiles
    """
    if ens_mean:
        truth_calib = np.mean(truth_calib, axis=-1)
        cnn_lower_calib = np.mean(cnn_lower_calib, axis=-1)
        cnn_upper_calib = np.mean(cnn_upper_calib, axis=-1)

    alpha = 1.0 - config.calibration_quantile
    quantile_corrections = []

    for var_idx in range(3):
        E_var = np.maximum(
            cnn_lower_calib[:, :, var_idx, ...] - truth_calib[:, :, var_idx, ...],
            truth_calib[:, :, var_idx, ...] - cnn_upper_calib[:, :, var_idx, ...],
        )

        quantile_correction_var = calculate_empirical_quantile(E_var, alpha=alpha)
        quantile_corrections.append(quantile_correction_var)

    return np.stack(quantile_corrections, axis=-1)


def apply_symmetric_quantile_adjustments(
    cnn_lower_test,
    cnn_upper_test,
    quantile_correction,
    ens_mean: bool = True,
):
    if ens_mean:
        cnn_lower_test = np.mean(cnn_lower_test, axis=(-1))
        cnn_upper_test = np.mean(cnn_upper_test, axis=(-1))
        quantile_correction_expanded = quantile_correction[None, ..., None]
    else:
        quantile_correction_expanded = quantile_correction[None, ..., None, None]
    cnn_lower_adjusted = cnn_lower_test - quantile_correction_expanded
    cnn_upper_adjusted = cnn_upper_test + quantile_correction_expanded

    return cnn_lower_adjusted, cnn_upper_adjusted


def check_quantile_coverage(
    test_set, lower_quantiles, upper_quantiles, ens_mean: bool = True
):
    """
    Checks coverage of quantile intervals.
    """
    if ens_mean:
        test_mean = np.mean(test_set, axis=-1)
    else:
        test_mean = test_set

    var_coverage = []
    for var in range(3):
        var_coverage.append(
            (test_mean[:, :, var] >= lower_quantiles[:, :, var])
            & (test_mean[:, :, var] <= upper_quantiles[:, :, var])
        )
    return np.stack(var_coverage, axis=2)


def visualize_quantile_coverage(coverage, sequence_name):
    """
    Visualize coverage performance over time.
    """
    # Average over grid points and seeds
    coverage_mean = np.mean(coverage, axis=(-1, 0))  # Shape: (timesteps, 3)
    coverage_std = np.std(np.mean(coverage, axis=-1), axis=0)  # Shape: (timesteps, 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        timesteps = range(len(coverage_mean))

        ax.plot(
            timesteps, coverage_mean[:, i], "b-", linewidth=2, label="Actual Coverage"
        )
        ax.fill_between(
            timesteps,
            coverage_mean[:, i] - coverage_std[:, i],
            coverage_mean[:, i] + coverage_std[:, i],
            alpha=0.3,
            color="blue",
            label="±1 Std Dev",
        )

        ax.axhline(
            y=config.calibration_quantile,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Target ({config.calibration_quantile:.0%})",
        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Coverage")
        ax.set_title(f"{var_name} Coverage over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()

    base_name = sequence_name.replace(".npz", "")
    save_path = f"{viz_dir}/{base_name}_quantile_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Quantile coverage visualization saved to: {save_path}")


def visualize_quantile_intervals(lower_quantiles, upper_quantiles, sequence_name):
    """
    Visualize quantile interval widths over time.
    """
    lower_mean = np.mean(lower_quantiles, axis=(0, -1))  # Average over seeds, grid
    upper_mean = np.mean(upper_quantiles, axis=(0, -1))
    interval_widths = upper_mean - lower_mean  # Shape: (timesteps, 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        timesteps = range(len(interval_widths))
        ax.plot(
            timesteps, interval_widths[:, i], "b-", linewidth=2, label="Interval Width"
        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Interval Width")
        ax.set_title(f"{var_name} Quantile Interval Width")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_name = sequence_name.replace(".npz", "")
    save_path = f"{viz_dir}/{base_name}_quantile_intervals.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Quantile interval visualization saved to: {save_path}")


def visualize_quantile_gridpoints(
    lower_quantiles,
    upper_quantiles,
    truth,
    qpens,
    sequence_name,
    random_seed=1,
    timestep=50,
):
    """
    Visualize quantile intervals at specific seed and timestep.
    """
    random_seed = min(random_seed, len(truth) - 1)
    timestep = min(timestep, len(truth[0]) - 1)

    truth_mean = np.mean(truth[random_seed, timestep], axis=-1)
    qpens_mean = np.mean(qpens[random_seed, timestep], axis=-1)
    lower_mean = lower_quantiles[random_seed, timestep]
    upper_mean = upper_quantiles[random_seed, timestep]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        gridpoints = range(len(truth_mean[i]))

        ax.fill_between(
            gridpoints,
            lower_mean[i],
            upper_mean[i],
            alpha=0.3,
            color="lightblue",
            label="Quantile Interval",
        )

        ax.plot(
            gridpoints, truth_mean[i], "go-", markersize=3, label="Truth", alpha=0.8
        )
        ax.plot(
            gridpoints, qpens_mean[i], "bs-", markersize=3, label="QPEns", alpha=0.8
        )
        ax.plot(
            gridpoints,
            (lower_mean[i] + upper_mean[i]) / 2,
            "r^-",
            markersize=3,
            label="Quantile Midpoint",
            alpha=0.8,
        )

        ax.set_xlabel("Grid Point")
        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name} at Seed {random_seed}, Timestep {timestep}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_name = sequence_name.replace(".npz", "")
    save_path = (
        f"{viz_dir}/{base_name}_quantile_gridpoints_seed{random_seed}_t{timestep}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Quantile gridpoint visualization saved to: {save_path}")


if __name__ == "__main__":
    app()
