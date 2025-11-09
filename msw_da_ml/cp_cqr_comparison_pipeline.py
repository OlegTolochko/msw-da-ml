import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from msw_da_ml.conformal_prediction.cp_data_generation import load_histories as load_cp_histories
from msw_da_ml.conformal_quantile_regression.cqr_data_generation import (
    load_histories as load_qr_histories,
)
from msw_da_ml.conformal_prediction.conformal_prediction import calibrate, check_coverage
from msw_da_ml.conformal_quantile_regression.cqr_prediction import (
    calibrate_quantile_intervals_symmetric,
    apply_symmetric_quantile_adjustments,
    check_quantile_coverage,
)
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config
viz_dir = get_output_dir(global_config.visualizations_out_filename)


def generate_comparison_analysis(
    cp_hist_name: str, cqr_hist_name: str, normalize_cp: bool = True, include_cnn_std: bool = False
):
    """
    Generate comparison plots between CP, normalized CP, and CQR methods.
    """
    # CP data
    cp_histories = load_cp_histories(cp_hist_name)
    cp_truth = np.asarray([h.truth for h in cp_histories])
    cp_qpens = np.asarray([h.qpens_analysis for h in cp_histories])
    cp_cnn = np.asarray([h.cnn_analysis for h in cp_histories])

    # CQR data
    cqr_histories = load_qr_histories(cqr_hist_name)
    cqr_truth = np.asarray([h.truth for h in cqr_histories])
    cqr_qpens = np.asarray([h.qpens_analysis for h in cqr_histories])
    cqr_lower = np.asarray([h.cnn_analysis_lower_quantiles for h in cqr_histories])
    cqr_upper = np.asarray([h.cnn_analysis_upper_quantiles for h in cqr_histories])

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
    cp_quantiles = calibrate(cp_qpens_calib, cp_cnn_calib, normalization_term=1)
    cp_cnn_mean = np.mean(cp_cnn_test, axis=-1)
    cp_quantiles_exp = np.tile(
        np.expand_dims(cp_quantiles, (0, -1)),
        (cp_cnn_mean.shape[0], 1, 1, cp_cnn_mean.shape[-1]),
    )
    cp_upper = cp_cnn_mean + cp_quantiles_exp
    cp_lower = cp_cnn_mean - cp_quantiles_exp
    cp_coverage = check_coverage(cp_qpens_test, cp_upper, cp_lower)

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

        cnn_std_coverage = check_coverage(cp_qpens, cnn_std_upper_intervals, cnn_std_lower_intervals)

    # Run CQR
    cqr_adjustments = calibrate_quantile_intervals_symmetric(
        cqr_qpens_calib, cqr_lower_calib, cqr_upper_calib
    )
    cqr_lower_adj, cqr_upper_adj = apply_symmetric_quantile_adjustments(
        cqr_lower_test, cqr_upper_test, cqr_adjustments
    )
    print(cqr_lower_adj.shape)
    cqr_coverage = check_quantile_coverage(cqr_qpens_test, cqr_lower_adj, cqr_upper_adj)
    print(cqr_coverage.shape)

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

    plot_coverage_comparison(coverages, methods, f"{cp_hist_name}_vs_{cqr_hist_name}")
    plot_interval_width_comparison(
        intervals, methods, f"{cp_hist_name}_vs_{cqr_hist_name}"
    )


def plot_coverage_comparison(coverages, method_names, save_name):
    """
    Creates 3x3 coverage + std comparison plot (3 variables x 3 methods).
    """
    fig, axes = plt.subplots(3, len(method_names), figsize=(5 * len(method_names), 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, var_name in enumerate(variable_names):
        for j, (coverage, method_name) in enumerate(zip(coverages, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]

            print(f"{save_name}, {method_names}: {coverage.shape}")

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
    save_path = f"{viz_dir}{save_name}_coverage_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Coverage comparison saved to: {save_path}")


def plot_interval_width_comparison(intervals, method_names, save_name):
    """
    Creates 3x3 interval width + std comparison plot (3 variables x 3 methods).
    """
    fig, axes = plt.subplots(3, len(method_names), figsize=(5 * len(method_names), 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    for i, var_name in enumerate(variable_names):
        for j, ((lower, upper), method_name) in enumerate(zip(intervals, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]

            interval_widths = upper - lower

            widths_mean = np.mean(interval_widths[:, :, i, :], axis=(0, -1))

            # Std over seeds
            widths_std = np.std(
                np.mean(interval_widths[:, :, i, :], axis=-1), axis=0
            )  # Shape: (timesteps,)

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

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Interval Width")
            ax.set_title(f"{var_name} - {method_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{viz_dir}{save_name}_interval_width_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Interval width comparison saved to: {save_path}")
