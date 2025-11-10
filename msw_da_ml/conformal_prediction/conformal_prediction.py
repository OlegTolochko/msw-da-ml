import numpy as np
from sklearn.model_selection import train_test_split
from cyclopts import App
import matplotlib.pyplot as plt

from msw_da_ml.conformal_prediction.cp_data_generation import load_histories
from msw_da_ml.settings import load_settings, get_output_dir

app = App()

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config

viz_dir = get_output_dir(global_config.visualizations_out_filename)


@app.command()
def conformal_prediction(cp_hist_name: str, normalize: bool = False, num_iterations: int = 10):
    """
    Runs conformal prediction pipeline.
    """
    histories = load_histories(cp_hist_name)
    truth_hist = np.asarray([history.truth for history in histories])
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_hist = np.asarray([history.cnn_analysis for history in histories])

    coverages = []
    for i in range(num_iterations):
        truth_calib, truth_test, qpens_calib, qpens_test, cnn_calib, cnn_test = (
            train_test_split(
                truth_hist,
                qpens_hist,
                cnn_hist,
                test_size=1 - config.calibration_split_ratio,
                random_state=config.calibration_split_seed + i,
            )
        )

        variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

        normalization_term = 1
        # Normalization based on variable-wise std
        if normalize:
            cnn_std = np.std(cnn_calib, axis=-1)
            # Epsilon is only required for rain, which can be 0
            cnn_std[:, :, 2] += config.rain_normalization_eps
            normalization_term = cnn_std
            for j, var_name in enumerate(variable_names):
                print(f"{var_name} mean std: {np.mean(cnn_std[:, :, j])}")
                print(f"{var_name} min std: {np.min(cnn_std[:, :, j])}")
                print(f"{var_name} max std: {np.max(cnn_std[:, :, j])}")

        # Since CNN is trained on qpens predictions, qpens is the truth for the CNN
        quantiles = calibrate(
            truth_calib=qpens_calib,
            cnn_calib=cnn_calib,
            normalization_term=normalization_term,
        )
        cnn_test_ens_mean = np.mean(cnn_test, axis=-1)

        normalization_term_test = 1.0
        if normalize:
            cnn_test_std = np.std(cnn_test, axis=-1)
            cnn_test_std[:, :, 2] += config.rain_normalization_eps
            normalization_term_test = cnn_test_std

        quantiles_expanded = (
            np.tile(
                np.expand_dims(quantiles, (0, -1)),
                (cnn_test_ens_mean.shape[0], 1, 1, cnn_test_ens_mean.shape[-1]),
            )
            * normalization_term_test
        )

        upper_intervals = cnn_test_ens_mean + quantiles_expanded
        lower_intervals = cnn_test_ens_mean - quantiles_expanded

        coverage = check_coverage(qpens_test, upper_intervals, lower_intervals)
        coverages.append(coverage)
        if normalize:
            cp_hist_name += "_normalized"

        quantiles = np.mean(quantiles_expanded, axis=(0, -1))
        if i == 0: # only visualize quantile intervals and gridpoint coverage for first iteration
            visualize_quantile_intervals(quantiles, cp_hist_name)
            visualize_coverage_gridpoints(
                upper_intervals, lower_intervals, truth_test, qpens_test, cnn_test, cp_hist_name
            )
    print(
        f"Target coverage: {config.calibration_quantile:.0%}"
    )
    mean_coverage = np.mean(coverages)
    var_coverage = np.var([np.mean(coverage) for coverage in coverages])
    print(f"Mean Coverage: {mean_coverage} for {num_iterations} data splits")
    print(f"Coverage Variance: {var_coverage} for {num_iterations} data splits")

    coverage_iter_mean = np.mean(coverages, axis=0)
    visualize_coverage(coverage_iter_mean, cp_hist_name)

@app.command()
def cnn_std_cov(cp_hist_name: str):
    histories = load_histories(cp_hist_name)
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_hist = np.asarray([history.cnn_analysis for history in histories])

    cnn_ens_mean = np.mean(cnn_hist, axis=-1)
    cnn_ens_std = np.std(cnn_hist, axis=-1)
    upper_intervals = cnn_ens_mean + cnn_ens_std
    lower_intervals = cnn_ens_mean - cnn_ens_std

    coverage = check_coverage(qpens_hist, upper_intervals, lower_intervals)
    mean_coverage = np.mean(coverage)
    print(f"Mean Coverage: {mean_coverage}")
    visualize_coverage(coverage, cp_hist_name + "_cnn_std")


@app.command()
def mcdo_std_cov(mcdo_hist_name: str):
    from msw_da_ml.conformal_prediction.mcdo_data_generation import (
        load_histories as load_mcdo_histories,
    )

    histories = load_mcdo_histories(mcdo_hist_name)

    qpens_hist = np.asarray([history.qpens_analysis for history in histories])

    cnn_mcdo_hist = np.asarray([history.cnn_analysis for history in histories])

    cnn_mcdo_ens_mean = np.mean(cnn_mcdo_hist, axis=-2)

    cnn_mcdo_mean = np.mean(cnn_mcdo_ens_mean, axis=-1)
    cnn_mcdo_std = np.std(cnn_mcdo_hist, axis=(-1,-2))

    upper_intervals = cnn_mcdo_mean + cnn_mcdo_std
    lower_intervals = cnn_mcdo_mean - cnn_mcdo_std

    coverage = check_coverage(qpens_hist, upper_intervals, lower_intervals)
    mean_coverage = np.mean(coverage)

    print(f"MCDO +-1 std coverage vs QPEns mean: {mean_coverage}")
    visualize_coverage(coverage, mcdo_hist_name + "_mcdo_std")


def check_coverage(
    test_set: np.ndarray, upper_quantiles: np.ndarray, lower_quantiles: np.ndarray
):
    """
    Checks whether test set is inside lower and upper intervals over ensemble dimension.
    """
    test_ens_mean = np.mean(test_set, axis=-1)
    coverage = (test_ens_mean > lower_quantiles) & (test_ens_mean < upper_quantiles)

    return coverage


def calibrate(truth_calib: np.ndarray, cnn_calib: np.ndarray, normalization_term):
    """
    Calculates calibration quantiles for conformal prediction.

    Returns:
        np.ndarray: Quantiles for each variable. Shape: (num_timesteps, 3)
    """
    # caluclate means for ensemble dim (-1)
    truth_ens_mean = np.mean(truth_calib, axis=(-1))
    cnn_ens_mean = np.mean(cnn_calib, axis=(-1))

    cnn_truth_mean_diff = np.abs(truth_ens_mean - cnn_ens_mean) / normalization_term
    print(np.mean(cnn_truth_mean_diff[:, :, 0]))
    print(np.mean(cnn_truth_mean_diff[:, :, 1]))
    print(np.mean(cnn_truth_mean_diff[:, :, 2]))

    # wind, water height and rain quantiles for each timestep
    quantiles = np.quantile(
        cnn_truth_mean_diff, q=config.calibration_quantile, axis=(0, -1)
    )  # shape: (num_timesteps, 3)

    return quantiles


def visualize_coverage(coverage: np.ndarray, hist_name: str):
    # shape (num_seeds, num_timesteps, 3, 250) -> (num_timesteps, 3)
    coverage_gridpoint_mean = np.mean(coverage, axis=-1)
    coverage_seed_gridpoint_mean = np.mean(coverage_gridpoint_mean, axis=(0))
    coverage_seed_gridpoint_std = np.std(coverage_gridpoint_mean, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        timesteps = range(coverage_seed_gridpoint_mean.shape[0])
        ax.plot(
            timesteps,
            coverage_seed_gridpoint_mean[:, i],
            "b-",
            linewidth=2,
            label="Actual Coverage",
        )
        ax.fill_between(
            timesteps,
            coverage_seed_gridpoint_mean[:, i] - coverage_seed_gridpoint_std[:, i],
            coverage_seed_gridpoint_mean[:, i] + coverage_seed_gridpoint_std[:, i],
            facecolor="blue",
            alpha=0.3,
            label="+-1 Standard Deviation",
        )
        ax.axhline(
            y=config.calibration_quantile,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Target Coverage ({config.calibration_quantile:.0%})",
        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Coverage")
        ax.set_title(f"{var_name} Coverage over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(
            np.min(coverage_seed_gridpoint_mean - coverage_seed_gridpoint_std) * 0.9, 1
        )

    plt.tight_layout()

    base_name = hist_name.replace(".npz", "")
    save_path = f"{viz_dir}/{base_name}_conformal_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Coverage visualization saved to: {save_path}")


def visualize_quantile_intervals(quantiles: np.ndarray, hist_name: str):
    interval_length = 2 * quantiles

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        timesteps = range(interval_length.shape[0])
        ax.plot(
            timesteps, interval_length[:, i], "b-", linewidth=2, label="Interval Length"
        )

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Interval Length")
        ax.set_title(f"{var_name} Interval Length over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(np.min(interval_length) * 0.95, np.max(interval_length) * 1.05)

    plt.tight_layout()

    base_name = hist_name.replace(".npz", "")
    save_path = f"{viz_dir}{base_name}_conformal_intervals.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Interval visualization saved to: {save_path}")


def visualize_coverage_gridpoints(
    upper_interval,
    lower_interval,
    truth,
    qpens,
    cnn,
    hist_name: str,
    random_seed: int = 10,
    timestep: int = 50,
):
    """
    Visualizes coverage performance for a set random seed and timestamp
    """
    truth_ens_mean = np.mean(truth, axis=-1)
    qpens_ens_mean = np.mean(qpens, axis=-1)
    cnn_ens_mean = np.mean(cnn, axis=-1)

    upper_interval_seed_timestep = upper_interval[random_seed, timestep]
    lower_interval_seed_timestep = lower_interval[random_seed, timestep]
    truth_seed_timestep = truth_ens_mean[random_seed, timestep]
    qpens_seed_timestep = qpens_ens_mean[random_seed, timestep]
    cnn_seed_timestep = cnn_ens_mean[random_seed, timestep]
    cnn_std_seed_timestep = np.std(cnn[random_seed, timestep], axis=-1)

    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        gridpoints = range(upper_interval.shape[-1])

        ax.fill_between(
            gridpoints,
            lower_interval_seed_timestep[i, :],
            upper_interval_seed_timestep[i, :],
            facecolor="lightblue",
            alpha=0.5,
            label="Confidence Interval",
        )

        ax.fill_between(
            gridpoints,
            cnn_seed_timestep[i, :] - cnn_std_seed_timestep[i, :],
            cnn_seed_timestep[i, :] + cnn_std_seed_timestep[i, :],
            facecolor="red",
            alpha=0.5,
            label="CNN +-1 Std",
        )

        ax.plot(
            gridpoints,
            truth_seed_timestep[i, :],
            "go-",
            markersize=2,
            linewidth=1.5,
            label="Truth",
            alpha=0.5,
            markerfacecolor="white",
            markeredgewidth=0.75,
        )

        ax.plot(
            gridpoints,
            qpens_seed_timestep[i, :],
            "bs-",
            markersize=2,
            linewidth=1.5,
            label="QPens",
            alpha=0.5,
            markerfacecolor="white",
            markeredgewidth=0.75,
        )

        ax.plot(
            gridpoints,
            cnn_seed_timestep[i, :],
            "r^-",
            markersize=2,
            linewidth=1.5,
            label="CNN",
            alpha=0.5,
            markerfacecolor="white",
            markeredgewidth=0.75,
        )

        ax.set_xlabel("Grid Point")
        ax.set_ylabel(var_name)
        ax.set_title(f"{var_name} at Seed {random_seed}, Timestep {timestep}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        y_range = ax.get_ylim()
        y_padding = (y_range[1] - y_range[0]) * 0.1
        ax.set_ylim(y_range[0] - y_padding, y_range[1] + y_padding)

    plt.tight_layout()

    base_name = hist_name.replace(".npz", "")
    save_path = (
        f"{viz_dir}{base_name}_conformal_gridpoints_seed{random_seed}_t{timestep}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Gridpoint visualization saved to: {save_path}")


def plot_non_conformity_scores(
    non_conformity_scores, quantiles, random_seed: int = 10, timestep: int = 50
):
    scores = non_conformity_scores[random_seed, timestep]
    quantiles_timestep = quantiles[timestep]

    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        ax.hist(
            scores[i, :],
            bins=30,
            alpha=0.7,
            color="blue",
            edgecolor="black",
            label=f"{var_name} Non conformity Scores",
        )

        ax.set_xlabel("Non conformity Score")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"{var_name} non conformity score for Seed {random_seed}, Timestep {timestep}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_name = "non_conformity_plot"
    save_path = f"{viz_dir}{base_name}_gridpoints_seed{random_seed}_t{timestep}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    app()
