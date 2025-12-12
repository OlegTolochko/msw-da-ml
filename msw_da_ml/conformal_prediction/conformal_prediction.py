import numpy as np
from sklearn.model_selection import train_test_split
from cyclopts import App
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


from msw_da_ml.conformal_prediction.cp_data_generation import load_histories
from msw_da_ml.settings import load_settings, get_output_dir
from msw_da_ml.conformal_prediction.mcdo_data_generation import (
    load_histories as load_mcdo_histories,
)
from msw_da_ml.evidential_regression.er_nig_data_generation import (
    load_histories as load_nig_histories,
)

app = App()

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config

viz_dir = get_output_dir(global_config.visualizations_out_filename)


@app.command()
def conformal_prediction(
    cp_hist_name: str, normalize: bool = False, num_iterations: int = 10
):
    """
    Runs conformal prediction pipeline.
    """
    histories = load_histories(cp_hist_name)
    truth_hist = np.asarray([history.truth for history in histories])
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_hist = np.asarray([history.cnn_analysis for history in histories])

    if normalize:
        cp_hist_name += "_normalized"

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
        quantiles, coverage, upper_intervals, lower_intervals = cp_main(
            cnn_calib, qpens_calib, cnn_test, qpens_test, normalize
        )

        coverages.append(coverage)
        if (
            i == 0
        ):  # only visualize quantile intervals and gridpoint coverage for first iteration
            visualize_quantile_intervals(quantiles, cp_hist_name)
            visualize_coverage_gridpoints(
                upper_intervals,
                lower_intervals,
                truth_test,
                qpens_test,
                cnn_test,
                cp_hist_name,
            )
    print(f"Target coverage: {config.calibration_quantile:.0%}")
    mean_coverage = np.mean(coverages)
    var_coverage = np.var([np.mean(coverage) for coverage in coverages])
    print(f"Mean Coverage: {mean_coverage} for {num_iterations} data splits")
    print(f"Coverage Variance: {var_coverage} for {num_iterations} data splits")

    coverage_iter_mean = np.mean(coverages, axis=0)
    visualize_coverage(coverage_iter_mean, cp_hist_name)


def cp_main(
    cnn_calib,
    qpens_calib,
    cnn_test,
    qpens_test,
    normalize,
    external_norm_calib=None,
    external_norm_test=None,
):
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    normalization_term = 1
    normalization_term_test = 1.0
    if normalize:
        height_eps = 0.01
        if external_norm_calib is not None:
            normalization_term = external_norm_calib
            normalization_term_test = external_norm_test
            normalization_term[:, :, 2] += 1e-6
            normalization_term_test[:, :, 2] += 1e-6
            normalization_term[:, :, 1] += height_eps
            normalization_term_test[:, :, 1] += height_eps
        else:
            # Ensemble std if no external norm
            cnn_std = np.std(cnn_calib, axis=-1)
            cnn_std[:, :, 2] += config.rain_normalization_eps
            normalization_term = cnn_std

            cnn_test_std = np.std(cnn_test, axis=-1)
            cnn_test_std[:, :, 2] += config.rain_normalization_eps
            normalization_term_test = cnn_test_std

    # Since CNN is trained on qpens predictions, qpens is the truth for the CNN
    quantiles = calibrate(
        truth_calib=qpens_calib,
        cnn_calib=cnn_calib,
        normalization_term=normalization_term,
    )
    cnn_test_ens_mean = np.mean(cnn_test, axis=-1)

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

    quantiles = np.mean(quantiles_expanded, axis=(0, -1))
    return quantiles, coverage, upper_intervals, lower_intervals


@app.command()
def cnn_std_cov(cp_hist_name: str):
    histories = load_histories(cp_hist_name)
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_hist = np.asarray([history.cnn_analysis for history in histories])

    cnn_ens_mean = np.mean(cnn_hist, axis=-1)
    cnn_ens_std = np.std(cnn_hist, axis=-1)

    alpha = 1 - config.calibration_quantile
    lower = norm.ppf(alpha / 2)
    upper = norm.ppf(1 - alpha / 2)
    print(lower)
    print(upper)

    upper_intervals = cnn_ens_mean + upper * cnn_ens_std
    lower_intervals = cnn_ens_mean + lower * cnn_ens_std

    coverage = check_coverage(qpens_hist, upper_intervals, lower_intervals)
    mean_coverage = np.mean(coverage)
    print(f"Mean Coverage: {mean_coverage}")
    visualize_coverage(coverage, cp_hist_name + "_cnn_std")


@app.command()
def mcdo_cp(
    mcdo_hist_name: str,
    cp_normalized: bool = False,
    ens_mean: bool = True,
    epistemic_norm: bool = False,
):
    histories = load_mcdo_histories(mcdo_hist_name)

    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    truth_hist = np.asarray([history.truth for history in histories])
    cnn_mean_mcdo_hist = np.asarray(
        [history.cnn_analysis_mean for history in histories]
    )
    cnn_logvar_mcdo_hist = np.asarray(
        [history.cnn_analysis_logvar for history in histories]
    )
    cnn_logvar_mcdo_hist = cnn_logvar_mcdo_hist.transpose(0, 1, 3, 4, 5, 2)

    epistemic_var = np.var(cnn_mean_mcdo_hist, axis=-2)
    aleatoric_var = np.mean(np.exp(cnn_logvar_mcdo_hist), axis=-2)
    total_std = np.sqrt(epistemic_var + aleatoric_var)

    cnn_mcdo_mean = np.mean(cnn_mean_mcdo_hist, axis=-2)

    (
        truth_calib,
        truth_test,
        qpens_calib,
        qpens_test,
        cnn_calib,
        cnn_test,
        std_calib,
        std_test,
        epistemic_calib,
        epistemic_test,
        aleatoric_calib,
        alaetoric_test,
    ) = train_test_split(
        truth_hist,
        qpens_hist,
        cnn_mcdo_mean,
        total_std,
        epistemic_var,
        aleatoric_var,
        test_size=1 - config.calibration_split_ratio,
        random_state=config.calibration_split_seed,
    )
    if cp_normalized:
        mcdo_hist_name += "_normalized"
        if epistemic_norm:
            external_norm_calib = np.mean(np.sqrt(epistemic_calib), axis=-1)
            external_norm_test = np.mean(np.sqrt(epistemic_test), axis=-1)
        else:
            external_norm_calib = np.mean(std_calib, axis=-1)
            external_norm_test = np.mean(std_test, axis=-1)
    else:
        external_norm_calib = None
        external_norm_test = None

    quantiles, coverage, upper_intervals, lower_intervals = cp_main(
        cnn_calib=cnn_calib,
        qpens_calib=qpens_calib,
        cnn_test=cnn_test,
        qpens_test=qpens_test,
        normalize=cp_normalized,
        external_norm_calib=external_norm_calib,
        external_norm_test=external_norm_test,
    )
    visualize_coverage(coverage, hist_name=mcdo_hist_name)

    if ens_mean:
        epistemic_test = epistemic_test.mean(axis=-1)
        alaetoric_test = alaetoric_test.mean(axis=-1)

    visualize_coverage_gridpoints(
        upper_intervals,
        lower_intervals,
        truth_test,
        qpens_test,
        cnn_test,
        mcdo_hist_name,
    )

    total_uncertainty = epistemic_test + alaetoric_test

    failure_mask = 1 - coverage.astype(int)

    failure_flat = failure_mask.flatten()
    epistemic_flat = epistemic_test.flatten()
    aleatoric_flat = alaetoric_test.flatten()
    total_flat = total_uncertainty.flatten()

    auroc_epistemic = roc_auc_score(failure_flat, epistemic_flat)
    auroc_aleatoric = roc_auc_score(failure_flat, aleatoric_flat)
    auroc_total = roc_auc_score(failure_flat, total_flat)

    print(f"OOD Detection (Prediction of Non-Coverage) AUROC:")
    print(f"  Epistemic: {auroc_epistemic:.4f}")
    print(f"  Aleatoric: {auroc_aleatoric:.4f}")
    print(f"  Total:     {auroc_total:.4f}")


@app.command()
def mcdo_std(mcdo_hist_name: str, ens_mean: bool = False):
    histories = load_mcdo_histories(mcdo_hist_name)

    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    truth_hist = np.asarray([history.truth for history in histories])
    cnn_mean_mcdo_hist = np.asarray(
        [history.cnn_analysis_mean for history in histories]
    )
    cnn_logvar_mcdo_hist = np.asarray(
        [history.cnn_analysis_logvar for history in histories]
    )

    if ens_mean:
        cnn_mean_mcdo_hist = np.mean(cnn_mean_mcdo_hist, axis=-2)

        cnn_vars = np.exp(cnn_logvar_mcdo_hist)
        cnn_vars_mean = np.mean(cnn_vars, axis=-2)

        epistemic_var = np.var(cnn_mean_mcdo_hist, axis=-1)
        aleatoric_var = np.mean(cnn_vars_mean, axis=-1)

        cnn_mcdo_mean = np.mean(cnn_mean_mcdo_hist, axis=-1)
        total_std = np.sqrt(epistemic_var + aleatoric_var)

    else:
        epistemic_var = np.var(cnn_mean_mcdo_hist, axis=-1)
        aleatoric_var = np.mean(np.exp(cnn_logvar_mcdo_hist), axis=-1)
        total_std = np.sqrt(epistemic_var + aleatoric_var)

        cnn_mcdo_mean = np.mean(cnn_mean_mcdo_hist, axis=-1)

    alpha = 1 - config.calibration_quantile
    z_score = norm.ppf(1 - alpha / 2)

    print(f"Target Coverage: {config.calibration_quantile:.0%}")
    print(f"Z-score used: {z_score:.4f}")

    upper_intervals = cnn_mcdo_mean + z_score * total_std
    lower_intervals = cnn_mcdo_mean - z_score * total_std

    coverage = check_coverage(
        qpens_hist, upper_intervals, lower_intervals, ens_mean=ens_mean
    )

    mean_coverage = np.mean(coverage)
    print(f"Mean Coverage: {mean_coverage:.4f}")

    visualize_coverage(coverage, mcdo_hist_name + "_mcdo_std")
    visualize_coverage_gridpoints(
        upper_intervals,
        lower_intervals,
        truth_hist,
        qpens_hist,
        cnn_mcdo_mean,
        mcdo_hist_name + "_mcdo_std",
        cnn_std=total_std,
    )


@app.command()
def nig_std(nig_hist_name: str, ens_mean: bool = False, max_std_clip: float = 10.0):
    """
    Compute prediction intervals using NIG model uncertainty estimates.
    """
    histories = load_nig_histories(nig_hist_name)

    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    truth_hist = np.asarray([history.truth for history in histories])
    cnn_gamma = np.asarray([history.cnn_analysis_gamma for history in histories])
    cnn_nu = np.asarray([history.cnn_analysis_nu for history in histories])
    cnn_alpha = np.asarray([history.cnn_analysis_alpha for history in histories])
    cnn_beta = np.asarray([history.cnn_analysis_beta for history in histories])

    print(f"NIG Parameter Statistics:")
    print(
        f"gamma: mean={np.mean(cnn_gamma):.4f}, std={np.std(cnn_gamma):.4f}, "
        f"min={np.min(cnn_gamma):.4f}, max={np.max(cnn_gamma):.4f}"
    )
    print(
        f"nu: mean={np.mean(cnn_nu):.4f}, std={np.std(cnn_nu):.4f}, "
        f"min={np.min(cnn_nu):.4f}, max={np.max(cnn_nu):.4f}"
    )
    print(
        f"  alpha: mean={np.mean(cnn_alpha):.4f}, std={np.std(cnn_alpha):.4f}, "
        f"min={np.min(cnn_alpha):.4f}, max={np.max(cnn_alpha):.4f}"
    )
    print(
        f"beta: mean={np.mean(cnn_beta):.4f}, std={np.std(cnn_beta):.4f}, "
        f"min={np.min(cnn_beta):.4f}, max={np.max(cnn_beta):.4f}"
    )

    if ens_mean:
        cnn_gamma = np.mean(cnn_gamma, axis=-1)
        cnn_nu = np.mean(cnn_nu, axis=-1)
        cnn_alpha = np.mean(cnn_alpha, axis=-1)
        cnn_beta = np.mean(cnn_beta, axis=-1)

    alpha_safe = np.maximum(cnn_alpha, 1.0 + 1e-6)
    nu_safe = np.maximum(cnn_nu, 1e-6)

    aleatoric_var = cnn_beta / (alpha_safe - 1.0)
    epistemic_var = aleatoric_var / nu_safe
    total_var = aleatoric_var + epistemic_var

    total_var_clipped = np.clip(total_var, 0, max_std_clip**2)
    total_std = np.sqrt(total_var_clipped)

    print(f"\nVariance Statistics (before clipping):")
    print(
        f"aleatoric_var: mean={np.mean(aleatoric_var):.4f}, max={np.max(aleatoric_var):.4f}"
    )
    print(
        f"epistemic_var: mean={np.mean(epistemic_var):.4f}, max={np.max(epistemic_var):.4f}"
    )
    print(
        f"total_std: mean={np.mean(np.sqrt(total_var)):.4f}, max={np.max(np.sqrt(total_var)):.4f}"
    )

    num_clipped = np.sum(total_var > max_std_clip**2)
    total_elements = total_var.size
    if num_clipped > 0:
        print(
            f"\n{num_clipped}/{total_elements} ({100 * num_clipped / total_elements:.2f}%) "
            f"variance values were clipped to max_std={max_std_clip}"
        )

    alpha = 1 - config.calibration_quantile
    z_score = norm.ppf(1 - alpha / 2)

    print(f"\nTarget Coverage: {config.calibration_quantile:.0%}")
    print(f"Z-score used: {z_score:.4f}")

    upper_intervals = cnn_gamma + z_score * total_std
    lower_intervals = cnn_gamma - z_score * total_std

    interval_lengths = upper_intervals - lower_intervals
    print(f"\nInterval Length Statistics:")
    print(f"mean={np.mean(interval_lengths):.4f}, std={np.std(interval_lengths):.4f}")
    print(f"min={np.min(interval_lengths):.4f}, max={np.max(interval_lengths):.4f}")

    coverage = check_coverage(
        qpens_hist, upper_intervals, lower_intervals, ens_mean=ens_mean
    )

    mean_coverage = np.mean(coverage)
    print(f"\nMean Coverage: {mean_coverage:.4f}")

    visualize_coverage(coverage, nig_hist_name + "_nig_std")
    visualize_coverage_gridpoints(
        upper_intervals,
        lower_intervals,
        truth_hist,
        qpens_hist,
        cnn_gamma,
        nig_hist_name + "_nig_std",
        cnn_std=total_std,
    )


def check_coverage(
    test_set: np.ndarray,
    upper_quantiles: np.ndarray,
    lower_quantiles: np.ndarray,
    ens_mean: bool = True,
):
    """
    Checks whether test set is inside lower and upper intervals over ensemble dimension.
    """
    if ens_mean:
        test = np.mean(test_set, axis=-1)
    else:
        test = test_set
    coverage = (test > lower_quantiles) & (test < upper_quantiles)

    return coverage


def calibrate(
    truth_calib: np.ndarray,
    cnn_calib: np.ndarray,
    normalization_term,
    ens_mean: bool = True,
):
    """
    Calculates calibration quantiles for conformal prediction.

    Returns:
        np.ndarray: Quantiles for each variable. Shape: (num_timesteps, 3)
    """
    if ens_mean:
        # caluclate means for ensemble dim (-1)
        truth_calib = np.mean(truth_calib, axis=(-1))
        cnn_calib = np.mean(cnn_calib, axis=(-1))

    cnn_truth_mean_diff = np.abs(truth_calib - cnn_calib) / normalization_term
    print(f"Mean diff u: {np.mean(cnn_truth_mean_diff[:, :, 0]):.6f}")
    print(f"Mean diff h: {np.mean(cnn_truth_mean_diff[:, :, 1]):.6f}")
    print(f"Mean diff r: {np.mean(cnn_truth_mean_diff[:, :, 2]):.6f}")

    # wind, water height and rain quantiles for each timestep
    if ens_mean:
        quantile_axes = (0, -1)
    else:
        quantile_axes = (0, -2, -1)
    quantiles = np.quantile(
        cnn_truth_mean_diff, q=config.calibration_quantile, axis=quantile_axes
    )  # shape: (num_timesteps, 3)

    return quantiles


def visualize_coverage(coverage: np.ndarray, hist_name: str):
    # shape (num_seeds, num_timesteps, 3, 250) -> (num_timesteps, 3)
    if coverage.ndim == 5:
        coverage_gridpoint_mean = np.mean(coverage, axis=(-2, -1))
    else:
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
    save_path = f"{viz_dir}/{base_name}_conformal_intervals.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Interval visualization saved to: {save_path}")


def visualize_coverage_gridpoints(
    upper_interval,
    lower_interval,
    truth,
    qpens,
    cnn,
    hist_name: str,
    cnn_std: np.ndarray = None,
    random_seed: int = 1,
    timestep: int = 50,
):
    """
    Visualizes coverage performance for a set random seed and timestamp
    """
    if truth.ndim == 5:
        truth_ens_mean = np.mean(truth, axis=-1)
    else:
        truth_ens_mean = truth

    if qpens.ndim == 5:
        qpens_ens_mean = np.mean(qpens, axis=-1)
    else:
        qpens_ens_mean = qpens

    if cnn.ndim == 5:
        cnn_ens_mean = np.mean(cnn, axis=-1)
        if cnn_std is None:
            cnn_std = np.std(cnn, axis=-1)
    else:
        cnn_ens_mean = cnn
        if cnn_std is None:
            cnn_std = np.zeros_like(cnn)

    # Handle ensemble dimension for intervals
    if upper_interval.ndim == 5:
        upper_interval = np.mean(upper_interval, axis=-1)
    if lower_interval.ndim == 5:
        lower_interval = np.mean(lower_interval, axis=-1)

    upper_interval_seed_timestep = upper_interval[random_seed, timestep]
    lower_interval_seed_timestep = lower_interval[random_seed, timestep]
    truth_seed_timestep = truth_ens_mean[random_seed, timestep]
    qpens_seed_timestep = qpens_ens_mean[random_seed, timestep]
    cnn_seed_timestep = cnn_ens_mean[random_seed, timestep]

    if cnn_std.ndim == 5:
        cnn_std = np.mean(cnn_std, axis=-1)
    cnn_std_seed_timestep = cnn_std[random_seed, timestep]

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
        f"{viz_dir}/{base_name}_conformal_gridpoints_seed{random_seed}_t{timestep}.png"
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
