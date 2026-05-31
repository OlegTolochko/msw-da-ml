import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import joblib
from datetime import datetime
from scipy.stats import norm

from msw_da_ml.data.evaluation_sequences import (
    load_evaluation_sequences as load_cp_sequences,
)
from msw_da_ml.inference.cqr_sequence import (
    load_cqr_evaluation_sequences,
)
from msw_da_ml.inference.mcdo_sequence import (
    load_mcdo_evaluation_sequences,
)
from msw_da_ml.uncertainty.split_cp import (
    calibrate,
    check_coverage,
    get_rf_norm,
    cp_main,
)
from msw_da_ml.uncertainty.rf_normalizer import get_rf_model_path
from msw_da_ml.uncertainty.cqr import (
    calibrate_quantile_intervals_symmetric,
    apply_symmetric_quantile_adjustments,
    check_quantile_coverage,
)
from msw_da_ml.inference.evidential_sequence import (
    load_nig_evaluation_sequences,
)
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config
viz_dir = get_output_dir(global_config.visualizations_out_filename)


def generate_comparison_analysis(
    cp_sequence_name: str,
    cqr_sequence_name: str,
    mcdo_sequence_name: str = "",
    nig_sequence_name: str = "",
    normalize_cp: bool = True,
    include_cnn_std: bool = False,
    include_rf: bool = False,
    ens_mean: bool = True,
):
    """
    Generate comparison plots between CP, normalized CP, CQR, and UQ baselines.
    """
    # CP data
    cp_histories = load_cp_sequences(cp_sequence_name)
    cp_truth = np.asarray([h.truth for h in cp_histories])
    cp_qpens = np.asarray([h.qpens_analysis for h in cp_histories])
    cp_cnn = np.asarray([h.cnn_analysis for h in cp_histories])

    # CQR data
    cqr_histories = load_cqr_evaluation_sequences(cqr_sequence_name)
    cqr_truth = np.asarray([h.truth for h in cqr_histories])
    cqr_qpens = np.asarray([h.qpens_analysis for h in cqr_histories])
    cqr_lower = np.asarray([h.cnn_analysis_lower_quantiles for h in cqr_histories])
    cqr_upper = np.asarray([h.cnn_analysis_upper_quantiles for h in cqr_histories])

    if nig_sequence_name:
        nig_sequences = load_nig_evaluation_sequences(nig_sequence_name)
        nig_qpens_hist = np.asarray(
            [sequence.qpens_analysis for sequence in nig_sequences]
        )
        nig_cnn_gamma = np.asarray(
            [sequence.cnn_analysis_gamma for sequence in nig_sequences]
        )
        nig_cnn_nu = np.asarray(
            [sequence.cnn_analysis_nu for sequence in nig_sequences]
        )
        nig_cnn_alpha = np.asarray(
            [sequence.cnn_analysis_alpha for sequence in nig_sequences]
        )
        nig_cnn_beta = np.asarray(
            [sequence.cnn_analysis_beta for sequence in nig_sequences]
        )

    # Split size
    random_state = config.calibration_split_seed
    test_size = 1 - config.calibration_split_ratio

    # CP splits
    (
        cp_truth_calib,
        cp_truth_test,
        cp_qpens_calib,
        cp_qpens_test,
        cp_cnn_calib,
        cp_cnn_test,
    ) = train_test_split(
        cp_truth, cp_qpens, cp_cnn, test_size=test_size, random_state=random_state
    )

    # CQR splits
    (
        cqr_truth_calib,
        cqr_truth_test,
        cqr_qpens_calib,
        cqr_qpens_test,
        cqr_lower_calib,
        cqr_lower_test,
        cqr_upper_calib,
        cqr_upper_test,
    ) = train_test_split(
        cqr_truth,
        cqr_qpens,
        cqr_lower,
        cqr_upper,
        test_size=test_size,
        random_state=random_state,
    )

    # Run CP (standard)
    cp_quantiles = calibrate(
        cp_qpens_calib, cp_cnn_calib, normalization_term=1, ens_mean=ens_mean
    )
    cp_cnn_mean = np.mean(cp_cnn_test, axis=-1)
    if ens_mean:
        cp_quantiles_exp = np.tile(
            np.expand_dims(cp_quantiles, (0, -1)),
            (cp_cnn_mean.shape[0], 1, 1, cp_cnn_mean.shape[-1]),
        )
        cp_upper = cp_cnn_mean + cp_quantiles_exp
        cp_lower = cp_cnn_mean - cp_quantiles_exp
    else:
        cp_quantiles_exp = np.tile(
            np.expand_dims(cp_quantiles, (0, -2, -1)),
            (cp_cnn_test.shape[0], 1, 1, cp_cnn_test.shape[-2], cp_cnn_test.shape[-1]),
        )
        cp_upper = cp_cnn_test + cp_quantiles_exp
        cp_lower = cp_cnn_test - cp_quantiles_exp

    cp_coverage = check_coverage(cp_qpens_test, cp_upper, cp_lower, ens_mean=ens_mean)

    # Run CP (normalized)
    if normalize_cp:
        cp_std = np.std(cp_cnn_calib, axis=-1)
        cp_std[:, :, 2] += config.rain_normalization_eps
        cp_norm_quantiles = calibrate(cp_qpens_calib, cp_cnn_calib, cp_std)
        cp_test_std = np.std(cp_cnn_test, axis=-1)
        cp_test_std[:, :, 2] += config.rain_normalization_eps
        cp_norm_quantiles_exp = (
            np.tile(
                np.expand_dims(cp_norm_quantiles, (0, -1)),
                (cp_cnn_mean.shape[0], 1, 1, cp_cnn_mean.shape[-1]),
            )
            * cp_test_std
        )
        cp_norm_upper = cp_cnn_mean + cp_norm_quantiles_exp
        cp_norm_lower = cp_cnn_mean - cp_norm_quantiles_exp
        cp_norm_coverage = check_coverage(cp_qpens_test, cp_norm_upper, cp_norm_lower)

    if include_cnn_std:
        cnn_ens_mean = np.mean(cp_cnn, axis=-1)
        cnn_ens_std = np.std(cp_cnn, axis=-1)
        cnn_std_upper_intervals = cnn_ens_mean + cnn_ens_std
        cnn_std_lower_intervals = cnn_ens_mean - cnn_ens_std

        cnn_std_coverage = check_coverage(
            cp_qpens, cnn_std_upper_intervals, cnn_std_lower_intervals
        )

    # Run CQR
    cqr_adjustments = calibrate_quantile_intervals_symmetric(
        cqr_qpens_calib, cqr_lower_calib, cqr_upper_calib, ens_mean=ens_mean
    )
    cqr_lower_adj, cqr_upper_adj = apply_symmetric_quantile_adjustments(
        cqr_lower_test, cqr_upper_test, cqr_adjustments, ens_mean=ens_mean
    )
    cqr_coverage = check_quantile_coverage(
        cqr_qpens_test, cqr_lower_adj, cqr_upper_adj, ens_mean=ens_mean
    )

    # Run MCDO STD
    if mcdo_sequence_name:
        mcdo_histories = load_mcdo_evaluation_sequences(mcdo_sequence_name)
        mcdo_qpens = np.asarray([h.qpens_analysis for h in mcdo_histories])
        mcdo_cnn_mean = np.asarray([h.cnn_analysis_mean for h in mcdo_histories])
        mcdo_cnn_logvar = np.asarray([h.cnn_analysis_logvar for h in mcdo_histories])
        mcdo_cnn_mean_raw = mcdo_cnn_mean
        mcdo_cnn_logvar_raw = mcdo_cnn_logvar

        if ens_mean:
            mcdo_cnn_mean = np.mean(mcdo_cnn_mean, axis=-2)
            mcdo_cnn_vars = np.mean(np.exp(mcdo_cnn_logvar), axis=-2)
        else:
            mcdo_cnn_vars = np.exp(mcdo_cnn_logvar)

        epistemic_var = np.var(mcdo_cnn_mean, axis=-1)
        aleatoric_var = np.mean(mcdo_cnn_vars, axis=-1)
        total_std = np.sqrt(epistemic_var + aleatoric_var)

        mcdo_mean_pred = np.mean(mcdo_cnn_mean, axis=-1)

        alpha = 1 - config.calibration_quantile
        z_score = norm.ppf(1 - alpha / 2)

        mcdo_upper = mcdo_mean_pred + z_score * total_std
        mcdo_lower = mcdo_mean_pred - z_score * total_std

        mcdo_coverage = check_coverage(
            mcdo_qpens, mcdo_upper, mcdo_lower, ens_mean=ens_mean
        )

    # Run evidential STD
    if nig_sequence_name:
        if ens_mean:
            nig_cnn_gamma_mean = np.mean(nig_cnn_gamma, axis=-1)
            nig_cnn_nu_mean = np.mean(nig_cnn_nu, axis=-1)
            nig_cnn_alpha_mean = np.mean(nig_cnn_alpha, axis=-1)
            nig_cnn_beta_mean = np.mean(nig_cnn_beta, axis=-1)
        else:
            nig_cnn_gamma_mean = nig_cnn_gamma
            nig_cnn_nu_mean = nig_cnn_nu
            nig_cnn_alpha_mean = nig_cnn_alpha
            nig_cnn_beta_mean = nig_cnn_beta

        alpha_safe = np.maximum(nig_cnn_alpha_mean, 1.0 + 1e-6)
        nu_safe = np.maximum(nig_cnn_nu_mean, 1e-6)

        aleatoric_var = nig_cnn_beta_mean / (alpha_safe - 1.0)
        epistemic_var = aleatoric_var / nu_safe
        total_var = aleatoric_var + epistemic_var

        max_std_clip = 10.0
        total_var_clipped = np.clip(total_var, 0, max_std_clip**2)
        nig_total_std = np.sqrt(total_var_clipped)

        alpha = 1 - config.calibration_quantile
        z_score = norm.ppf(1 - alpha / 2)

        nig_upper = nig_cnn_gamma_mean + z_score * nig_total_std
        nig_lower = nig_cnn_gamma_mean - z_score * nig_total_std

        nig_coverage = check_coverage(
            nig_qpens_hist, nig_upper, nig_lower, ens_mean=ens_mean
        )

    # Run RF Normalized CP
    if include_rf:
        rf_path = get_rf_model_path()
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"RF model not found at {rf_path}.")
        rf = joblib.load(rf_path)

        rf_norm_calib = get_rf_norm(cp_cnn_calib, rf)
        rf_norm_test = get_rf_norm(cp_cnn_test, rf)

        if ens_mean:
            rf_norm_calib = np.mean(rf_norm_calib, axis=-1)
            rf_norm_test = np.mean(rf_norm_test, axis=-1)

        _, rf_coverage, rf_upper, rf_lower = cp_main(
            cp_cnn_calib,
            cp_qpens_calib,
            cp_cnn_test,
            cp_qpens_test,
            normalize=True,
            external_norm_calib=rf_norm_calib,
            external_norm_test=rf_norm_test,
            ens_mean=ens_mean,
        )

    # Comparison plots:
    methods = ["CP", "CQR"]
    coverages = [cp_coverage, cqr_coverage]
    intervals = [(cp_lower, cp_upper), (cqr_lower_adj, cqr_upper_adj)]
    if normalize_cp:
        methods.append("CP (Normalized)")
        coverages.append(cp_norm_coverage)
        intervals.append((cp_norm_lower, cp_norm_upper))

    if include_cnn_std:
        methods.append("CNN STD")
        coverages.append(cnn_std_coverage)
        intervals.append((cnn_std_lower_intervals, cnn_std_upper_intervals))

    if mcdo_sequence_name:
        methods.append("MCDO STD")
        coverages.append(mcdo_coverage)
        intervals.append((mcdo_lower, mcdo_upper))

    if nig_sequence_name:
        methods.append("Evidential STD")
        coverages.append(nig_coverage)
        intervals.append((nig_lower, nig_upper))

    if include_rf:
        methods.append("RF Normalized CP")
        coverages.append(rf_coverage)
        intervals.append((rf_lower, rf_upper))

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    save_name = f"uq_method_comparison_{timestamp}"

    plot_coverage_comparison(coverages, methods, save_name, ens_mean=ens_mean)
    plot_interval_width_comparison(intervals, methods, save_name, ens_mean=ens_mean)
    plot_interval_width_log_comparison(
        intervals, methods, save_name, ens_mean=ens_mean
    )

    plot_coverage_and_width_joint(
        coverages,
        intervals,
        methods,
        save_name,
        ens_mean=ens_mean,
    )

    plot_uq_model_rmse_comparison(
        cqr_qpens,
        cqr_lower,
        cqr_upper,
        mcdo_qpens if mcdo_sequence_name else None,
        mcdo_cnn_mean_raw if mcdo_sequence_name else None,
        nig_qpens_hist if nig_sequence_name else None,
        nig_cnn_gamma if nig_sequence_name else None,
        save_name,
    )

    if mcdo_sequence_name or nig_sequence_name:
        plot_uncertainty_decomposition(
            mcdo_cnn_mean_raw if mcdo_sequence_name else None,
            mcdo_cnn_logvar_raw if mcdo_sequence_name else None,
            nig_cnn_nu if nig_sequence_name else None,
            nig_cnn_alpha if nig_sequence_name else None,
            nig_cnn_beta if nig_sequence_name else None,
            save_name,
            log_scale=False,
        )
        plot_uncertainty_decomposition(
            mcdo_cnn_mean_raw if mcdo_sequence_name else None,
            mcdo_cnn_logvar_raw if mcdo_sequence_name else None,
            nig_cnn_nu if nig_sequence_name else None,
            nig_cnn_alpha if nig_sequence_name else None,
            nig_cnn_beta if nig_sequence_name else None,
            save_name,
            log_scale=True,
        )


def plot_coverage_comparison(coverages, method_names, save_name, ens_mean: bool = True):
    """
    Creates coverage comparison plot (3 variables x N methods).
    Handles both ens_mean=True (4D coverage) and ens_mean=False (5D coverage).
    """
    fig, axes = plt.subplots(3, len(method_names), figsize=(5 * len(method_names), 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, var_name in enumerate(variable_names):
        for j, (coverage, method_name) in enumerate(zip(coverages, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]

            if coverage.ndim == 5:
                # Shape: (seeds, time, 3, grid, ens), average over grid and ens
                coverage_mean = np.mean(coverage[:, :, i, :, :], axis=(0, -2, -1))
                coverage_std = np.std(
                    np.mean(coverage[:, :, i, :, :], axis=(-2, -1)), axis=0
                )
            else:
                # Shape: (seeds, time, 3, grid), average over grid only
                coverage_mean = np.mean(coverage[:, :, i, :], axis=(0, -1))
                coverage_std = np.std(np.mean(coverage[:, :, i, :], axis=-1), axis=0)

            timesteps = range(len(coverage_mean))
            ax.plot(
                timesteps, coverage_mean, "b-", linewidth=2, label="Actual Coverage"
            )
            ax.fill_between(
                timesteps,
                coverage_mean - coverage_std,
                coverage_mean + coverage_std,
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
            ax.set_title(f"{var_name} - {method_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = f"{viz_dir}/{save_name}_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Coverage comparison saved to: {save_path}")
    plt.close(fig)


def plot_interval_width_comparison(
    intervals, method_names, save_name, ens_mean: bool = True
):
    """
    Creates interval width comparison plot (3 variables x N methods).
    Handles both ens_mean=True (4D intervals) and ens_mean=False (5D intervals).
    """
    fig, axes = plt.subplots(3, len(method_names), figsize=(5 * len(method_names), 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, var_name in enumerate(variable_names):
        for j, ((lower, upper), method_name) in enumerate(zip(intervals, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]

            interval_widths = upper - lower

            if interval_widths.ndim == 5:
                # Shape: (seeds, time, 3, grid, ens), average over grid and ens
                widths_mean = np.mean(interval_widths[:, :, i, :, :], axis=(0, -2, -1))
                widths_std = np.std(
                    np.mean(interval_widths[:, :, i, :, :], axis=(-2, -1)), axis=0
                )
            else:
                # Shape: (seeds, time, 3, grid), average over grid only
                widths_mean = np.mean(interval_widths[:, :, i, :], axis=(0, -1))
                widths_std = np.std(
                    np.mean(interval_widths[:, :, i, :], axis=-1), axis=0
                )

            timesteps = range(len(widths_mean))
            ax.plot(timesteps, widths_mean, "b-", linewidth=2, label="Interval Width")
            ax.fill_between(
                timesteps,
                widths_mean - widths_std,
                widths_mean + widths_std,
                alpha=0.3,
                color="blue",
                label="±1 Std Dev",
            )

            upper_display = widths_mean + widths_std
            upper_finite = upper_display[np.isfinite(upper_display)]
            if upper_finite.size:
                robust_top = np.percentile(upper_finite, 98) * 1.15
                raw_top = np.max(upper_finite) * 1.05
                if robust_top > 0 and raw_top > 2.5 * robust_top:
                    ax.set_ylim(0, robust_top)
                    ax.text(
                        0.02,
                        0.92,
                        f"axis clipped\nmax={raw_top / 1.05:.2g}",
                        transform=ax.transAxes,
                        fontsize=8,
                        va="top",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )
                else:
                    ax.set_ylim(0, raw_top)

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Interval Width")
            ax.set_title(f"{var_name} - {method_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{viz_dir}/{save_name}_interval_width.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Interval width comparison saved to: {save_path}")
    plt.close(fig)


def plot_interval_width_log_comparison(
    intervals, method_names, save_name, ens_mean: bool = True
):
    fig, axes = plt.subplots(3, len(method_names), figsize=(5 * len(method_names), 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, var_name in enumerate(variable_names):
        for j, ((lower, upper), method_name) in enumerate(zip(intervals, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]
            interval_widths = upper - lower

            if interval_widths.ndim == 5:
                widths_mean = np.mean(interval_widths[:, :, i, :, :], axis=(0, -2, -1))
                widths_std = np.std(
                    np.mean(interval_widths[:, :, i, :, :], axis=(-2, -1)), axis=0
                )
            else:
                widths_mean = np.mean(interval_widths[:, :, i, :], axis=(0, -1))
                widths_std = np.std(
                    np.mean(interval_widths[:, :, i, :], axis=-1), axis=0
                )

            finite_positive = widths_mean[np.isfinite(widths_mean) & (widths_mean > 0)]
            min_positive = (
                np.min(finite_positive) * 0.5 if finite_positive.size else 1e-12
            )
            band_lower = np.maximum(widths_mean - widths_std, min_positive)
            band_upper = np.maximum(widths_mean + widths_std, min_positive)
            plot_mean = np.maximum(widths_mean, min_positive)

            timesteps = range(len(widths_mean))
            ax.plot(timesteps, plot_mean, "b-", linewidth=2, label="Interval Width")
            ax.fill_between(
                timesteps,
                band_lower,
                band_upper,
                alpha=0.25,
                color="blue",
                label="±1 Std Dev",
            )

            ax.set_yscale("log")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Interval Width (log)")
            ax.set_title(f"{var_name} - {method_name}")
            ax.legend()
            ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_path = f"{viz_dir}/{save_name}_interval_width_log.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Log interval width comparison saved to: {save_path}")
    plt.close(fig)


def plot_coverage_and_width_joint(
    coverages,
    intervals,
    method_names,
    save_name,
    ens_mean: bool = True,
):
    """
    Joint plot: Coverage (left y-axis) + Interval width (right y-axis)
    for each variable and method. Layout: 3 rows (u,h,r) x N methods.
    """
    fig, axes = plt.subplots(
        3,
        len(method_names),
        figsize=(8 * len(method_names), 12),
        sharex=True,
        sharey=True,
    )
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, var_name in enumerate(variable_names):
        for j, (coverage, (lower, upper), method_name) in enumerate(
            zip(coverages, intervals, method_names)
        ):
            ax_cov = axes[i, j] if len(method_names) > 1 else axes[i]
            ax_w = ax_cov.twinx()

            # coverage stats
            if coverage.ndim == 5:
                # (seeds, time, 3, grid, ens)
                cov_mean = np.mean(coverage[:, :, i, :, :], axis=(0, -2, -1))
                cov_std = np.std(
                    np.mean(coverage[:, :, i, :, :], axis=(-2, -1)), axis=0
                )
            else:
                # (seeds, time, 3, grid)
                cov_mean = np.mean(coverage[:, :, i, :], axis=(0, -1))
                cov_std = np.std(np.mean(coverage[:, :, i, :], axis=-1), axis=0)

            # width stats
            widths = upper - lower
            if widths.ndim == 5:
                # (seeds, time, 3, grid, ens)
                w_mean = np.mean(widths[:, :, i, :, :], axis=(0, -2, -1))
                w_std = np.std(np.mean(widths[:, :, i, :, :], axis=(-2, -1)), axis=0)
            else:
                # (seeds, time, 3, grid)
                w_mean = np.mean(widths[:, :, i, :], axis=(0, -1))
                w_std = np.std(np.mean(widths[:, :, i, :], axis=-1), axis=0)

            timesteps = np.arange(len(cov_mean))

            # coverage
            ax_cov.plot(
                timesteps, cov_mean, color="tab:blue", linewidth=2, label="Coverage"
            )
            ax_cov.fill_between(
                timesteps,
                cov_mean - cov_std,
                cov_mean + cov_std,
                color="tab:blue",
                alpha=0.15,
                label="Coverage ±1σ",
            )
            ax_cov.axhline(
                y=config.calibration_quantile,
                color="tab:blue",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            ax_cov.set_ylim(0, 1)

            # width
            ax_w.plot(timesteps, w_mean, color="tab:orange", linewidth=2, label="Width")
            ax_w.fill_between(
                timesteps,
                w_mean - w_std,
                w_mean + w_std,
                color="tab:orange",
                alpha=0.15,
                label="Width ±1σ",
            )

            if i == 0:
                ax_cov.set_title(method_name)

            ax_cov.set_ylabel("Coverage")
            ax_w.set_ylabel("Width")
            ax_cov.grid(True, alpha=0.25)

            h1, l1 = ax_cov.get_legend_handles_labels()
            h2, l2 = ax_w.get_legend_handles_labels()
            ax_cov.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

            if i == 2:
                ax_cov.set_xlabel("Timestep")

            if j == 0:
                ax_cov.text(
                    -0.12,
                    0.5,
                    var_name,
                    transform=ax_cov.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                )

    plt.tight_layout()
    save_path = f"{viz_dir}/{save_name}_coverage_width_joint.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Coverage+Width joint plot saved to: {save_path}")
    plt.close(fig)


def _ensemble_mean(values: np.ndarray) -> np.ndarray:
    if values.ndim == 5:
        return np.mean(values, axis=-1)
    return values


def _rmse_over_time(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_mean = _ensemble_mean(target)
    error = prediction - target_mean
    return np.sqrt(np.mean(error**2, axis=(0, -1)))


def plot_uq_model_rmse_comparison(
    cqr_qpens,
    cqr_lower,
    cqr_upper,
    mcdo_qpens,
    mcdo_cnn_mean,
    nig_qpens,
    nig_cnn_gamma,
    save_name,
):
    models = []

    cqr_midpoint = 0.5 * (np.mean(cqr_lower, axis=-1) + np.mean(cqr_upper, axis=-1))
    models.append(("CQR midpoint", _rmse_over_time(cqr_midpoint, cqr_qpens)))

    if mcdo_qpens is not None and mcdo_cnn_mean is not None:
        mcdo_member_mean = np.mean(mcdo_cnn_mean, axis=-2)
        mcdo_mean = np.mean(mcdo_member_mean, axis=-1)
        models.append(("MCDO mean", _rmse_over_time(mcdo_mean, mcdo_qpens)))

    if nig_qpens is not None and nig_cnn_gamma is not None:
        nig_mean = np.mean(nig_cnn_gamma, axis=-1)
        models.append(("Evidential mean", _rmse_over_time(nig_mean, nig_qpens)))

    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    for var_idx, (ax, var_name) in enumerate(zip(axes, variable_names)):
        for model_name, rmse in models:
            timesteps = np.arange(rmse.shape[0])
            ax.plot(timesteps, rmse[:, var_idx], linewidth=2, label=model_name)
        ax.set_title(var_name)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_path = f"{viz_dir}/{save_name}_model_rmse.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"UQ model RMSE comparison saved to: {save_path}")
    plt.close(fig)


def _mean_uncertainty_by_time(values: np.ndarray) -> np.ndarray:
    return np.mean(values, axis=(0, -1))


def plot_uncertainty_decomposition(
    mcdo_cnn_mean,
    mcdo_cnn_logvar,
    nig_cnn_nu,
    nig_cnn_alpha,
    nig_cnn_beta,
    save_name,
    log_scale: bool = False,
):
    rows = []

    if mcdo_cnn_mean is not None and mcdo_cnn_logvar is not None:
        mcdo_mean_ens = np.mean(mcdo_cnn_mean, axis=-2)
        mcdo_vars_ens = np.mean(np.exp(mcdo_cnn_logvar), axis=-2)
        mcdo_epistemic = np.var(mcdo_mean_ens, axis=-1)
        mcdo_aleatoric = np.mean(mcdo_vars_ens, axis=-1)
        rows.append(
            (
                "MCDO",
                _mean_uncertainty_by_time(mcdo_aleatoric),
                _mean_uncertainty_by_time(mcdo_epistemic),
            )
        )

    if (
        nig_cnn_nu is not None
        and nig_cnn_alpha is not None
        and nig_cnn_beta is not None
    ):
        nig_nu = np.mean(nig_cnn_nu, axis=-1)
        nig_alpha = np.mean(nig_cnn_alpha, axis=-1)
        nig_beta = np.mean(nig_cnn_beta, axis=-1)

        alpha_safe = np.maximum(nig_alpha, 1.0 + 1e-6)
        nu_safe = np.maximum(nig_nu, 1e-6)
        nig_aleatoric = nig_beta / (alpha_safe - 1.0)
        nig_epistemic = nig_aleatoric / nu_safe
        rows.append(
            (
                "Evidential",
                _mean_uncertainty_by_time(nig_aleatoric),
                _mean_uncertainty_by_time(nig_epistemic),
            )
        )

    if not rows:
        return

    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(15, 4.5 * len(rows)), squeeze=False, sharex=True
    )

    for row_idx, (method_name, aleatoric, epistemic) in enumerate(rows):
        for var_idx, var_name in enumerate(variable_names):
            ax = axes[row_idx, var_idx]
            timesteps = np.arange(aleatoric.shape[0])
            ax.plot(timesteps, aleatoric[:, var_idx], linewidth=2, label="AU")
            ax.plot(timesteps, epistemic[:, var_idx], linewidth=2, label="EU")
            ax.set_title(f"{method_name} - {var_name}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Variance" if not log_scale else "Variance (log)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            upper = np.concatenate([aleatoric[:, var_idx], epistemic[:, var_idx]])
            upper = upper[np.isfinite(upper)]
            if log_scale:
                positive = upper[upper > 0]
                if positive.size:
                    ax.set_yscale("log")
                    ax.set_ylim(np.min(positive) * 0.5, np.max(positive) * 1.25)
                    ax.grid(True, alpha=0.3, which="both")
            elif upper.size:
                robust_top = np.percentile(upper, 98) * 1.15
                raw_top = np.max(upper) * 1.05
                if robust_top > 0 and raw_top > 2.5 * robust_top:
                    ax.set_ylim(0, robust_top)
                    ax.text(
                        0.02,
                        0.92,
                        f"axis clipped\nmax={raw_top / 1.05:.2g}",
                        transform=ax.transAxes,
                        fontsize=8,
                        va="top",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )
                else:
                    ax.set_ylim(0, raw_top)

    plt.tight_layout()
    suffix = (
        "uncertainty_decomposition_log"
        if log_scale
        else "uncertainty_decomposition"
    )
    save_path = f"{viz_dir}/{save_name}_{suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"UQ uncertainty decomposition saved to: {save_path}")
    plt.close(fig)
