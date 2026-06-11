import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
import os
import joblib
from datetime import datetime
from pathlib import Path
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
    check_coverage,
    get_rf_norm,
    cp_main,
)
from msw_da_ml.uncertainty.conformal_quantile import calculate_empirical_quantile
from msw_da_ml.uncertainty.rf_normalizer import get_rf_model_path
from msw_da_ml.uncertainty.cqr import (
    calibrate_quantile_intervals_symmetric,
    apply_symmetric_quantile_adjustments,
    check_quantile_coverage,
)
from msw_da_ml.inference.evidential_sequence import (
    load_nig_evaluation_sequences,
)
from msw_da_ml.inference.ensemble_sequence import (
    load_ensemble_evaluation_sequences,
)
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config
viz_dir = get_output_dir(global_config.visualizations_out_filename)
variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _finite_mean(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def _finite_width_time_stats(
    widths: np.ndarray, var_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    var_widths = np.array(widths[:, :, var_idx, ...], dtype=float, copy=True)
    var_widths[~np.isfinite(var_widths)] = np.nan

    if var_widths.ndim == 4:
        per_seed_time = np.nanmean(var_widths, axis=(-2, -1))
    else:
        per_seed_time = np.nanmean(var_widths, axis=-1)

    return np.nanmean(per_seed_time, axis=0), np.nanstd(per_seed_time, axis=0)


def write_average_set_size_table(
    intervals: list[tuple[np.ndarray, np.ndarray]],
    method_names: list[str],
    save_name: str,
):
    rows = []
    for method_name, (lower, upper) in zip(method_names, intervals):
        widths = upper - lower
        nan_count = int(np.isnan(widths).sum())
        inf_count = int(np.isinf(widths).sum())
        rows.append(
            (
                method_name,
                [_finite_mean(widths[:, :, var_idx, ...]) for var_idx in range(3)],
                nan_count,
                inf_count,
            )
        )

    table_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & $u$ & $h$ & $r$ & NaN & Inf \\\\",
        "\\midrule",
    ]
    for method_name, means, nan_count, inf_count in rows:
        table_lines.append(
            f"{method_name} & "
            f"{_format_float(means[0])} & "
            f"{_format_float(means[1])} & "
            f"{_format_float(means[2])} & "
            f"{nan_count} & {inf_count} \\\\"
        )
    table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Average set size (interval width) by method and variable. "
            "NaN and Inf columns report excluded non-finite interval-width values.}",
            "\\end{table}",
        ]
    )

    table_path = f"{viz_dir}/{save_name}_average_set_size_table.tex"
    with open(table_path, "w", encoding="utf-8") as table_file:
        table_file.write("\n".join(table_lines) + "\n")

    nonfinite_rows = [
        (method_name, nan_count, inf_count)
        for method_name, _, nan_count, inf_count in rows
        if nan_count or inf_count
    ]
    if nonfinite_rows:
        print("Non-finite average set size values excluded:")
        for method_name, nan_count, inf_count in nonfinite_rows:
            print(f"  {method_name}: NaN={nan_count}, Inf={inf_count}")
    else:
        print("No NaN/Inf interval-width values were excluded.")
    print(f"Average set size table saved to: {table_path}")

    return rows


def _build_seed_splits(num_seeds: int, num_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")
    if num_seeds < 2:
        raise ValueError("At least two sequence histories are needed for calibration/test splits")

    seed_indices = np.arange(num_seeds)
    test_size = 1 - config.calibration_split_ratio
    splits = []
    for split_idx in range(num_splits):
        calib_idx, test_idx = train_test_split(
            seed_indices,
            test_size=test_size,
            random_state=config.calibration_split_seed + split_idx,
        )
        splits.append((np.asarray(calib_idx), np.asarray(test_idx)))
    return splits


def _validate_seed_count(
    expected_num_seeds: int,
    method_name: str,
    values: np.ndarray,
):
    if values.shape[0] != expected_num_seeds:
        raise ValueError(
            f"{method_name} has {values.shape[0]} sequence histories, "
            f"expected {expected_num_seeds}. All methods must share the same "
            "seed dimension for comparable splits."
        )


def _seed_labels(histories) -> np.ndarray:
    return np.asarray([history.seed for history in histories])


def _validate_seed_labels(
    expected_seed_labels: np.ndarray,
    method_name: str,
    seed_labels: np.ndarray,
):
    if not np.array_equal(seed_labels, expected_seed_labels):
        raise ValueError(
            f"{method_name} seed labels/order do not match CP seed labels. "
            f"Expected {expected_seed_labels.tolist()}, got {seed_labels.tolist()}."
        )


def _quantile_expand_shape(quantiles: np.ndarray, target_ndim: int) -> tuple[int, ...]:
    return (1, quantiles.shape[0], quantiles.shape[1]) + (1,) * (target_ndim - 3)


def _calibrate_scaled_intervals(
    truth_calib: np.ndarray,
    center_calib: np.ndarray,
    scale_calib: np.ndarray,
    truth_test: np.ndarray,
    center_test: np.ndarray,
    scale_test: np.ndarray,
    ens_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ens_mean:
        truth_calib = np.mean(truth_calib, axis=-1)

    if truth_calib.shape != center_calib.shape:
        raise ValueError(
            f"Calibration truth shape {truth_calib.shape} does not match "
            f"center shape {center_calib.shape}"
        )
    if center_calib.shape != scale_calib.shape:
        raise ValueError(
            f"Calibration center shape {center_calib.shape} does not match "
            f"scale shape {scale_calib.shape}"
        )
    if center_test.shape != scale_test.shape:
        raise ValueError(
            f"Test center shape {center_test.shape} does not match "
            f"scale shape {scale_test.shape}"
        )

    scores = np.abs(truth_calib - center_calib) / np.maximum(scale_calib, 1e-12)
    alpha = 1.0 - config.calibration_quantile
    quantiles = []
    for var_idx in range(3):
        quantiles.append(
            calculate_empirical_quantile(
                scores[:, :, var_idx, ...],
                alpha=alpha,
            )
        )
    quantiles = np.stack(quantiles, axis=-1)
    quantiles_expanded = quantiles.reshape(_quantile_expand_shape(quantiles, center_test.ndim))

    upper = center_test + quantiles_expanded * scale_test
    lower = center_test - quantiles_expanded * scale_test
    coverage = check_coverage(truth_test, upper, lower, ens_mean=ens_mean)
    return coverage, lower, upper


def _run_scaled_interval_splits(
    truth: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ens_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverages = []
    lowers = []
    uppers = []
    for calib_idx, test_idx in splits:
        coverage, lower, upper = _calibrate_scaled_intervals(
            truth[calib_idx],
            center[calib_idx],
            scale[calib_idx],
            truth[test_idx],
            center[test_idx],
            scale[test_idx],
            ens_mean=ens_mean,
        )
        coverages.append(coverage)
        lowers.append(lower)
        uppers.append(upper)
    return (
        np.concatenate(coverages, axis=0),
        np.concatenate(lowers, axis=0),
        np.concatenate(uppers, axis=0),
    )


def _run_fixed_z_intervals(
    truth: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    ens_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = 1.0 - config.calibration_quantile
    z_score = norm.ppf(1.0 - alpha / 2.0)
    upper = center + z_score * scale
    lower = center - z_score * scale
    coverage = check_coverage(truth, upper, lower, ens_mean=ens_mean)
    return coverage, lower, upper


def _run_gaussian_intervals(
    truth: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ens_mean: bool,
    calibrate_gaussian: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if calibrate_gaussian:
        return _run_scaled_interval_splits(
            truth, center, scale, splits, ens_mean=ens_mean
        )
    return _run_fixed_z_intervals(truth, center, scale, ens_mean=ens_mean)


def _gaussian_method_name(base_name: str, calibrate_gaussian: bool) -> str:
    suffix = "Calibrated" if calibrate_gaussian else "Raw z"
    return f"{base_name} ({suffix})"


def _run_cp_splits(
    qpens: np.ndarray,
    cnn: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    normalize: bool,
    ens_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverages = []
    lowers = []
    uppers = []
    for calib_idx, test_idx in splits:
        _, coverage, upper, lower = cp_main(
            cnn[calib_idx],
            qpens[calib_idx],
            cnn[test_idx],
            qpens[test_idx],
            normalize=normalize,
            ens_mean=ens_mean,
        )
        coverages.append(coverage)
        lowers.append(lower)
        uppers.append(upper)
    return (
        np.concatenate(coverages, axis=0),
        np.concatenate(lowers, axis=0),
        np.concatenate(uppers, axis=0),
    )


def _run_cqr_splits(
    qpens: np.ndarray,
    lower_quantiles: np.ndarray,
    upper_quantiles: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ens_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverages = []
    lowers = []
    uppers = []
    for calib_idx, test_idx in splits:
        adjustments = calibrate_quantile_intervals_symmetric(
            qpens[calib_idx],
            lower_quantiles[calib_idx],
            upper_quantiles[calib_idx],
            ens_mean=ens_mean,
        )
        lower_adj, upper_adj = apply_symmetric_quantile_adjustments(
            lower_quantiles[test_idx],
            upper_quantiles[test_idx],
            adjustments,
            ens_mean=ens_mean,
        )
        coverage = check_quantile_coverage(
            qpens[test_idx], lower_adj, upper_adj, ens_mean=ens_mean
        )
        coverages.append(coverage)
        lowers.append(lower_adj)
        uppers.append(upper_adj)
    return (
        np.concatenate(coverages, axis=0),
        np.concatenate(lowers, axis=0),
        np.concatenate(uppers, axis=0),
    )


def _mcdo_center_scale(
    cnn_mean: np.ndarray, cnn_logvar: np.ndarray, ens_mean: bool
) -> tuple[np.ndarray, np.ndarray]:
    if ens_mean:
        sample_mean = np.mean(cnn_mean, axis=-2)
        sample_var = np.mean(np.exp(cnn_logvar), axis=-2)
    else:
        sample_mean = cnn_mean
        sample_var = np.exp(cnn_logvar)

    epistemic_var = np.var(sample_mean, axis=-1)
    aleatoric_var = np.mean(sample_var, axis=-1)
    center = np.mean(sample_mean, axis=-1)
    return center, np.sqrt(epistemic_var + aleatoric_var)


def _nig_center_scale(
    gamma: np.ndarray,
    nu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    ens_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if ens_mean:
        gamma = np.mean(gamma, axis=-1)
        nu = np.mean(nu, axis=-1)
        alpha = np.mean(alpha, axis=-1)
        beta = np.mean(beta, axis=-1)

    alpha_safe = np.maximum(alpha, 1.0 + 1e-6)
    nu_safe = np.maximum(nu, 1e-6)
    aleatoric_var = beta / (alpha_safe - 1.0)
    epistemic_var = aleatoric_var / nu_safe
    return gamma, np.sqrt(aleatoric_var + epistemic_var)


def _deep_ensemble_center_scale(
    cnn_mean: np.ndarray, cnn_logvar: np.ndarray, ens_mean: bool
) -> tuple[np.ndarray, np.ndarray]:
    if ens_mean:
        sample_mean = np.mean(cnn_mean, axis=-2)
        sample_var = np.mean(np.exp(cnn_logvar), axis=-2)
    else:
        sample_mean = cnn_mean
        sample_var = np.exp(cnn_logvar)

    epistemic_var = np.var(sample_mean, axis=-1)
    aleatoric_var = np.mean(sample_var, axis=-1)
    center = np.mean(sample_mean, axis=-1)
    return center, np.sqrt(epistemic_var + aleatoric_var)


def _single_model_center_scale(
    cnn_mean: np.ndarray,
    cnn_logvar: np.ndarray,
    model_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_stored_models = cnn_mean.shape[-1]
    if model_index < 0 or model_index >= num_stored_models:
        raise IndexError(
            f"model_index must be between 0 and {num_stored_models - 1}; "
            f"got {model_index}"
        )

    selected_mean = cnn_mean[..., model_index]
    selected_var = np.exp(cnn_logvar[..., model_index])
    epistemic_var = np.var(selected_mean, axis=-1)
    aleatoric_var = np.mean(selected_var, axis=-1)
    center = np.mean(selected_mean, axis=-1)
    return center, np.sqrt(epistemic_var + aleatoric_var)


def _single_model_output_name(
    sequence_name: str,
    model_index: int,
    output_name: str,
    calibrate_gaussian: bool,
) -> str:
    if output_name:
        return output_name.replace(".npz", "")
    stem = Path(sequence_name).stem
    mode = "calibrated" if calibrate_gaussian else "raw_z"
    return f"{stem}_model{model_index:02d}_single_model_gaussian_{mode}"


def generate_single_model_gaussian_analysis(
    ensemble_sequence_name: str,
    model_index: int = 0,
    num_splits: int = 10,
    calibrate_gaussian: bool = True,
    output_name: str = "",
):
    """Generate Gaussian interval plots from one stored model member.

    The selected model's physical ensemble axis is used as the Gaussian sample
    axis, while all saved sequence histories/seeds remain in the plot batch.
    """
    histories = load_ensemble_evaluation_sequences(ensemble_sequence_name)
    qpens = np.asarray([history.qpens_analysis for history in histories])
    ensemble_cnn_mean = np.asarray([history.cnn_analysis_mean for history in histories])
    ensemble_cnn_logvar = np.asarray(
        [history.cnn_analysis_logvar for history in histories]
    )

    if ensemble_cnn_mean.ndim != 6:
        raise ValueError(
            "Expected cnn_analysis_mean with shape "
            "(sequence, time, variable, grid, physical_ensemble, deep_ensemble_member); "
            f"got {ensemble_cnn_mean.shape}"
        )
    if qpens.shape != ensemble_cnn_mean.shape[:-1]:
        raise ValueError(
            f"QPEns shape {qpens.shape} does not match prediction shape "
            f"{ensemble_cnn_mean.shape[:-1]}"
        )

    center, scale = _single_model_center_scale(
        ensemble_cnn_mean, ensemble_cnn_logvar, model_index
    )
    splits = _build_seed_splits(qpens.shape[0], num_splits)
    coverage, lower, upper = _run_gaussian_intervals(
        qpens,
        center,
        scale,
        splits,
        ens_mean=True,
        calibrate_gaussian=calibrate_gaussian,
    )
    widths = upper - lower

    save_name = _single_model_output_name(
        ensemble_sequence_name, model_index, output_name, calibrate_gaussian
    )
    method_names = [_gaussian_method_name("Single Model Gaussian", calibrate_gaussian)]

    plot_coverage_comparison([coverage], method_names, save_name, ens_mean=True)
    plot_interval_width_comparison(
        [(lower, upper)], method_names, save_name, ens_mean=True
    )
    write_average_set_size_table([(lower, upper)], method_names, save_name)

    print(f"Sequences: {len(histories)}")
    print(f"Stored model index: {model_index}")
    print(f"Gaussian calibration: {calibrate_gaussian}")
    if calibrate_gaussian:
        print(f"Seed-only splits: {num_splits}")
    else:
        alpha = 1.0 - config.calibration_quantile
        print(f"Fixed z-score: {norm.ppf(1.0 - alpha / 2.0):.4f}")
    print(f"Target coverage: {config.calibration_quantile:.0%}")
    for var_idx, var_name in enumerate(variable_names):
        print(
            f"{var_name}: mean coverage={np.mean(coverage[:, :, var_idx, :]):.4f}, "
            f"average set size={np.nanmean(widths[:, :, var_idx, :]):.4f}"
        )

    return coverage, lower, upper


def generate_comparison_analysis(
    cp_sequence_name: str,
    cqr_sequence_name: str,
    mcdo_sequence_name: str = "",
    nig_sequence_name: str = "",
    ensemble_sequence_name: str = "",
    normalize_cp: bool = True,
    include_cnn_std: bool = False,
    include_rf: bool = False,
    ens_mean: bool = True,
    num_splits: int = 10,
    single_model_index: int | None = None,
    calibrate_gaussian: bool = True,
):
    """
    Generate comparison plots between CP, normalized CP, CQR, and UQ baselines.
    """
    # CP data
    cp_histories = load_cp_sequences(cp_sequence_name)
    cp_seed_labels = _seed_labels(cp_histories)
    cp_qpens = np.asarray([h.qpens_analysis for h in cp_histories])
    cp_cnn = np.asarray([h.cnn_analysis for h in cp_histories])

    # CQR data
    cqr_histories = load_cqr_evaluation_sequences(cqr_sequence_name)
    _validate_seed_labels(cp_seed_labels, "CQR", _seed_labels(cqr_histories))
    cqr_qpens = np.asarray([h.qpens_analysis for h in cqr_histories])
    cqr_lower = np.asarray([h.cnn_analysis_lower_quantiles for h in cqr_histories])
    cqr_upper = np.asarray([h.cnn_analysis_upper_quantiles for h in cqr_histories])

    if mcdo_sequence_name:
        mcdo_histories = load_mcdo_evaluation_sequences(mcdo_sequence_name)
        _validate_seed_labels(cp_seed_labels, "MCDO", _seed_labels(mcdo_histories))
        mcdo_qpens = np.asarray([h.qpens_analysis for h in mcdo_histories])
        mcdo_cnn_mean = np.asarray([h.cnn_analysis_mean for h in mcdo_histories])
        mcdo_cnn_logvar = np.asarray([h.cnn_analysis_logvar for h in mcdo_histories])
        mcdo_cnn_mean_raw = mcdo_cnn_mean
        mcdo_cnn_logvar_raw = mcdo_cnn_logvar

    if nig_sequence_name:
        nig_sequences = load_nig_evaluation_sequences(nig_sequence_name)
        _validate_seed_labels(cp_seed_labels, "Evidential", _seed_labels(nig_sequences))
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

    if ensemble_sequence_name:
        ensemble_histories = load_ensemble_evaluation_sequences(ensemble_sequence_name)
        _validate_seed_labels(
            cp_seed_labels, "Deep ensemble", _seed_labels(ensemble_histories)
        )
        ensemble_qpens = np.asarray([h.qpens_analysis for h in ensemble_histories])
        ensemble_cnn_mean = np.asarray(
            [h.cnn_analysis_mean for h in ensemble_histories]
        )
        ensemble_cnn_logvar = np.asarray(
            [h.cnn_analysis_logvar for h in ensemble_histories]
        )
        ensemble_cnn_mean_raw = ensemble_cnn_mean
        ensemble_cnn_logvar_raw = ensemble_cnn_logvar

    num_seed_histories = cp_qpens.shape[0]
    _validate_seed_count(num_seed_histories, "CQR", cqr_qpens)
    if mcdo_sequence_name:
        _validate_seed_count(num_seed_histories, "MCDO", mcdo_qpens)
    if nig_sequence_name:
        _validate_seed_count(num_seed_histories, "Evidential", nig_qpens_hist)
    if ensemble_sequence_name:
        _validate_seed_count(num_seed_histories, "Deep ensemble", ensemble_qpens)

    splits = _build_seed_splits(num_seed_histories, num_splits)

    # Run CP (standard) over the shared seed splits.
    cp_coverage, cp_lower, cp_upper = _run_cp_splits(
        cp_qpens, cp_cnn, splits, normalize=False, ens_mean=ens_mean
    )

    # Run CP (normalized)
    if normalize_cp:
        cp_norm_coverage, cp_norm_lower, cp_norm_upper = _run_cp_splits(
            cp_qpens, cp_cnn, splits, normalize=True, ens_mean=ens_mean
        )

    if include_cnn_std:
        if ens_mean:
            cnn_center = np.mean(cp_cnn, axis=-1)
            cnn_scale = np.std(cp_cnn, axis=-1)
        else:
            cnn_center = cp_cnn
            cnn_scale = np.broadcast_to(np.std(cp_cnn, axis=-1)[..., None], cp_cnn.shape)
        cnn_std_coverage, cnn_std_lower_intervals, cnn_std_upper_intervals = (
            _run_gaussian_intervals(
                cp_qpens,
                cnn_center,
                cnn_scale,
                splits,
                ens_mean=ens_mean,
                calibrate_gaussian=calibrate_gaussian,
            )
        )

    # Run CQR
    cqr_coverage, cqr_lower_adj, cqr_upper_adj = _run_cqr_splits(
        cqr_qpens, cqr_lower, cqr_upper, splits, ens_mean=ens_mean
    )

    # Run calibrated Gaussian intervals for stochastic/predictive UQ models.
    if mcdo_sequence_name:
        mcdo_center, mcdo_scale = _mcdo_center_scale(
            mcdo_cnn_mean, mcdo_cnn_logvar, ens_mean=ens_mean
        )
        mcdo_coverage, mcdo_lower, mcdo_upper = _run_gaussian_intervals(
            mcdo_qpens,
            mcdo_center,
            mcdo_scale,
            splits,
            ens_mean=ens_mean,
            calibrate_gaussian=calibrate_gaussian,
        )

    if nig_sequence_name:
        nig_center, nig_scale = _nig_center_scale(
            nig_cnn_gamma,
            nig_cnn_nu,
            nig_cnn_alpha,
            nig_cnn_beta,
            ens_mean=ens_mean,
        )
        nig_coverage, nig_lower, nig_upper = _run_gaussian_intervals(
            nig_qpens_hist,
            nig_center,
            nig_scale,
            splits,
            ens_mean=ens_mean,
            calibrate_gaussian=calibrate_gaussian,
        )

    if ensemble_sequence_name:
        ensemble_center, ensemble_scale = _deep_ensemble_center_scale(
            ensemble_cnn_mean, ensemble_cnn_logvar, ens_mean=ens_mean
        )
        ensemble_coverage, ensemble_lower, ensemble_upper = (
            _run_gaussian_intervals(
                ensemble_qpens,
                ensemble_center,
                ensemble_scale,
                splits,
                ens_mean=ens_mean,
                calibrate_gaussian=calibrate_gaussian,
            )
        )

        if single_model_index is not None:
            single_model_center, single_model_scale = _single_model_center_scale(
                ensemble_cnn_mean, ensemble_cnn_logvar, single_model_index
            )
            single_model_coverage, single_model_lower, single_model_upper = (
                _run_gaussian_intervals(
                    ensemble_qpens,
                    single_model_center,
                    single_model_scale,
                    splits,
                    ens_mean=True,
                    calibrate_gaussian=calibrate_gaussian,
                )
            )
    elif single_model_index is not None:
        raise ValueError("single_model_index requires an ensemble_sequence_name")

    # Run RF Normalized CP over the same shared splits.
    if include_rf:
        rf_path = get_rf_model_path()
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"RF model not found at {rf_path}.")
        rf = joblib.load(rf_path)

        rf_norm = get_rf_norm(cp_cnn, rf)
        if ens_mean:
            rf_norm = np.mean(rf_norm, axis=-1)
        rf_coverages = []
        rf_lowers = []
        rf_uppers = []
        for calib_idx, test_idx in splits:
            _, coverage, upper, lower = cp_main(
                cp_cnn[calib_idx],
                cp_qpens[calib_idx],
                cp_cnn[test_idx],
                cp_qpens[test_idx],
                normalize=True,
                external_norm_calib=rf_norm[calib_idx],
                external_norm_test=rf_norm[test_idx],
                ens_mean=ens_mean,
            )
            rf_coverages.append(coverage)
            rf_lowers.append(lower)
            rf_uppers.append(upper)
        rf_coverage = np.concatenate(rf_coverages, axis=0)
        rf_lower = np.concatenate(rf_lowers, axis=0)
        rf_upper = np.concatenate(rf_uppers, axis=0)

    # Comparison plots:
    methods = ["CP", "CQR"]
    coverages = [cp_coverage, cqr_coverage]
    intervals = [(cp_lower, cp_upper), (cqr_lower_adj, cqr_upper_adj)]
    if normalize_cp:
        methods.append("CP (Normalized)")
        coverages.append(cp_norm_coverage)
        intervals.append((cp_norm_lower, cp_norm_upper))

    if include_cnn_std:
        methods.append(_gaussian_method_name("CNN STD", calibrate_gaussian))
        coverages.append(cnn_std_coverage)
        intervals.append((cnn_std_lower_intervals, cnn_std_upper_intervals))

    if mcdo_sequence_name:
        methods.append(_gaussian_method_name("MCDO Gaussian", calibrate_gaussian))
        coverages.append(mcdo_coverage)
        intervals.append((mcdo_lower, mcdo_upper))

    if nig_sequence_name:
        methods.append(_gaussian_method_name("NIG Gaussian", calibrate_gaussian))
        coverages.append(nig_coverage)
        intervals.append((nig_lower, nig_upper))

    if ensemble_sequence_name:
        methods.append(
            _gaussian_method_name("Deep Ensemble Gaussian", calibrate_gaussian)
        )
        coverages.append(ensemble_coverage)
        intervals.append((ensemble_lower, ensemble_upper))

        if single_model_index is not None:
            methods.append(
                _gaussian_method_name("Single Model Gaussian", calibrate_gaussian)
            )
            coverages.append(single_model_coverage)
            intervals.append((single_model_lower, single_model_upper))

    if include_rf:
        methods.append("RF Normalized CP")
        coverages.append(rf_coverage)
        intervals.append((rf_lower, rf_upper))

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    save_name = f"uq_method_comparison_{timestamp}"

    plot_coverage_comparison(coverages, methods, save_name, ens_mean=ens_mean)
    plot_interval_width_comparison(intervals, methods, save_name, ens_mean=ens_mean)
    write_average_set_size_table(intervals, methods, save_name)

    plot_uq_model_rmse_comparison(
        cqr_qpens,
        cqr_lower,
        cqr_upper,
        mcdo_qpens if mcdo_sequence_name else None,
        mcdo_cnn_mean_raw if mcdo_sequence_name else None,
        nig_qpens_hist if nig_sequence_name else None,
        nig_cnn_gamma if nig_sequence_name else None,
        ensemble_qpens if ensemble_sequence_name else None,
        ensemble_cnn_mean_raw if ensemble_sequence_name else None,
        save_name,
    )

    if mcdo_sequence_name or nig_sequence_name or ensemble_sequence_name:
        plot_uncertainty_decomposition(
            mcdo_cnn_mean_raw if mcdo_sequence_name else None,
            mcdo_cnn_logvar_raw if mcdo_sequence_name else None,
            nig_cnn_nu if nig_sequence_name else None,
            nig_cnn_alpha if nig_sequence_name else None,
            nig_cnn_beta if nig_sequence_name else None,
            ensemble_cnn_mean_raw if ensemble_sequence_name else None,
            ensemble_cnn_logvar_raw if ensemble_sequence_name else None,
            save_name,
            log_scale=False,
        )
        plot_uncertainty_component(
            mcdo_cnn_mean_raw if mcdo_sequence_name else None,
            mcdo_cnn_logvar_raw if mcdo_sequence_name else None,
            nig_cnn_nu if nig_sequence_name else None,
            nig_cnn_alpha if nig_sequence_name else None,
            nig_cnn_beta if nig_sequence_name else None,
            ensemble_cnn_mean_raw if ensemble_sequence_name else None,
            ensemble_cnn_logvar_raw if ensemble_sequence_name else None,
            save_name,
            component="aleatoric",
            log_scale=False,
        )
        plot_uncertainty_component(
            mcdo_cnn_mean_raw if mcdo_sequence_name else None,
            mcdo_cnn_logvar_raw if mcdo_sequence_name else None,
            nig_cnn_nu if nig_sequence_name else None,
            nig_cnn_alpha if nig_sequence_name else None,
            nig_cnn_beta if nig_sequence_name else None,
            ensemble_cnn_mean_raw if ensemble_sequence_name else None,
            ensemble_cnn_logvar_raw if ensemble_sequence_name else None,
            save_name,
            component="epistemic",
            log_scale=False,
        )
        plot_error_vs_time_colored_by_eu(
            mcdo_qpens if mcdo_sequence_name else None,
            mcdo_cnn_mean_raw if mcdo_sequence_name else None,
            mcdo_cnn_logvar_raw if mcdo_sequence_name else None,
            nig_qpens_hist if nig_sequence_name else None,
            nig_cnn_gamma if nig_sequence_name else None,
            nig_cnn_nu if nig_sequence_name else None,
            nig_cnn_alpha if nig_sequence_name else None,
            nig_cnn_beta if nig_sequence_name else None,
            ensemble_qpens if ensemble_sequence_name else None,
            ensemble_cnn_mean_raw if ensemble_sequence_name else None,
            ensemble_cnn_logvar_raw if ensemble_sequence_name else None,
            save_name,
        )


def plot_coverage_comparison(coverages, method_names, save_name, ens_mean: bool = True):
    """
    Creates coverage comparison plot (3 variables x N methods).
    Handles both ens_mean=True (4D coverage) and ens_mean=False (5D coverage).
    """
    fig, axes = plt.subplots(3, len(method_names), figsize=(5 * len(method_names), 12))
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
    for i, var_name in enumerate(variable_names):
        for j, ((lower, upper), method_name) in enumerate(zip(intervals, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]

            interval_widths = upper - lower
            widths_mean, widths_std = _finite_width_time_stats(interval_widths, i)

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
    for i, var_name in enumerate(variable_names):
        for j, ((lower, upper), method_name) in enumerate(zip(intervals, method_names)):
            ax = axes[i, j] if len(method_names) > 1 else axes[i]
            interval_widths = upper - lower
            widths_mean, widths_std = _finite_width_time_stats(interval_widths, i)

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
            w_mean, w_std = _finite_width_time_stats(widths, i)

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
    ensemble_qpens,
    ensemble_cnn_mean,
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

    if ensemble_qpens is not None and ensemble_cnn_mean is not None:
        ensemble_member_mean = np.mean(ensemble_cnn_mean, axis=-2)
        ensemble_mean = np.mean(ensemble_member_mean, axis=-1)
        models.append(
            ("Deep ensemble mean", _rmse_over_time(ensemble_mean, ensemble_qpens))
        )

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
    finite_values = np.array(values, dtype=float, copy=True)
    finite_values[~np.isfinite(finite_values)] = np.nan
    return np.nanmean(finite_values, axis=(0, -1))


def _uncertainty_rows(
    mcdo_cnn_mean,
    mcdo_cnn_logvar,
    nig_cnn_nu,
    nig_cnn_alpha,
    nig_cnn_beta,
    ensemble_cnn_mean,
    ensemble_cnn_logvar,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
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

    if ensemble_cnn_mean is not None and ensemble_cnn_logvar is not None:
        ensemble_mean_ens = np.mean(ensemble_cnn_mean, axis=-2)
        ensemble_vars_ens = np.mean(np.exp(ensemble_cnn_logvar), axis=-2)
        ensemble_epistemic = np.var(ensemble_mean_ens, axis=-1)
        ensemble_aleatoric = np.mean(ensemble_vars_ens, axis=-1)
        rows.append(
            (
                "Deep Ensemble",
                _mean_uncertainty_by_time(ensemble_aleatoric),
                _mean_uncertainty_by_time(ensemble_epistemic),
            )
        )

    return rows


def _apply_uncertainty_axis_scale(ax, values: np.ndarray, log_scale: bool):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return

    if log_scale:
        positive = finite[finite > 0]
        if positive.size:
            ax.set_yscale("log")
            ax.set_ylim(np.min(positive) * 0.5, np.max(positive) * 1.25)
            ax.grid(True, alpha=0.3, which="both")
        return

    robust_top = np.percentile(finite, 98) * 1.15
    raw_top = np.max(finite) * 1.05
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
    elif raw_top > 0:
        ax.set_ylim(0, raw_top)


def plot_uncertainty_decomposition(
    mcdo_cnn_mean,
    mcdo_cnn_logvar,
    nig_cnn_nu,
    nig_cnn_alpha,
    nig_cnn_beta,
    ensemble_cnn_mean,
    ensemble_cnn_logvar,
    save_name,
    log_scale: bool = False,
):
    rows = _uncertainty_rows(
        mcdo_cnn_mean,
        mcdo_cnn_logvar,
        nig_cnn_nu,
        nig_cnn_alpha,
        nig_cnn_beta,
        ensemble_cnn_mean,
        ensemble_cnn_logvar,
    )

    if not rows:
        return

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
            _apply_uncertainty_axis_scale(ax, upper, log_scale)

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


def plot_uncertainty_component(
    mcdo_cnn_mean,
    mcdo_cnn_logvar,
    nig_cnn_nu,
    nig_cnn_alpha,
    nig_cnn_beta,
    ensemble_cnn_mean,
    ensemble_cnn_logvar,
    save_name,
    component: str,
    log_scale: bool = False,
):
    rows = _uncertainty_rows(
        mcdo_cnn_mean,
        mcdo_cnn_logvar,
        nig_cnn_nu,
        nig_cnn_alpha,
        nig_cnn_beta,
        ensemble_cnn_mean,
        ensemble_cnn_logvar,
    )
    if not rows:
        return
    if component not in {"aleatoric", "epistemic"}:
        raise ValueError("component must be 'aleatoric' or 'epistemic'")

    component_idx = 1 if component == "epistemic" else 0
    label = "EU" if component == "epistemic" else "AU"
    title = "Epistemic uncertainty" if component == "epistemic" else "Aleatoric uncertainty"

    fig, axes = plt.subplots(
        len(rows), 3, figsize=(15, 4.5 * len(rows)), squeeze=False, sharex=True
    )

    for row_idx, row in enumerate(rows):
        method_name = row[0]
        values = row[component_idx + 1]
        for var_idx, var_name in enumerate(variable_names):
            ax = axes[row_idx, var_idx]
            timesteps = np.arange(values.shape[0])
            ax.plot(timesteps, values[:, var_idx], linewidth=2, label=label)
            ax.set_title(f"{method_name} - {var_name}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Variance" if not log_scale else "Variance (log)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            _apply_uncertainty_axis_scale(ax, values[:, var_idx], log_scale)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    suffix = f"{component}_uncertainty"
    if log_scale:
        suffix += "_log"
    save_path = f"{viz_dir}/{save_name}_{suffix}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{title} plot saved to: {save_path}")
    plt.close(fig)


def _per_seed_time_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_mean = _ensemble_mean(target)
    error = prediction - target_mean
    return np.sqrt(np.mean(error**2, axis=-1))


def _eu_grid_values_from_mcdo(mean_values: np.ndarray) -> np.ndarray:
    member_mean = np.mean(mean_values, axis=-2)
    return np.var(member_mean, axis=-1)


def _eu_grid_values_from_evidential(
    nu_values: np.ndarray,
    alpha_values: np.ndarray,
    beta_values: np.ndarray,
) -> np.ndarray:
    nu = np.mean(nu_values, axis=-1)
    alpha = np.mean(alpha_values, axis=-1)
    beta = np.mean(beta_values, axis=-1)
    alpha_safe = np.maximum(alpha, 1.0 + 1e-6)
    nu_safe = np.maximum(nu, 1e-6)
    aleatoric = beta / (alpha_safe - 1.0)
    return aleatoric / nu_safe


def _mean_prediction_from_samples(mean_values: np.ndarray) -> np.ndarray:
    return np.mean(np.mean(mean_values, axis=-2), axis=-1)


def plot_error_vs_time_colored_by_eu(
    mcdo_qpens,
    mcdo_cnn_mean,
    mcdo_cnn_logvar,
    nig_qpens,
    nig_cnn_gamma,
    nig_cnn_nu,
    nig_cnn_alpha,
    nig_cnn_beta,
    ensemble_qpens,
    ensemble_cnn_mean,
    ensemble_cnn_logvar,
    save_name,
):
    rows = []
    if mcdo_qpens is not None and mcdo_cnn_mean is not None:
        rows.append(
            (
                "MCDO",
                _per_seed_time_rmse(
                    _mean_prediction_from_samples(mcdo_cnn_mean), mcdo_qpens
                ),
                np.nanmean(_eu_grid_values_from_mcdo(mcdo_cnn_mean), axis=-1),
            )
        )
    if (
        nig_qpens is not None
        and nig_cnn_gamma is not None
        and nig_cnn_nu is not None
        and nig_cnn_alpha is not None
        and nig_cnn_beta is not None
    ):
        rows.append(
            (
                "Evidential",
                _per_seed_time_rmse(np.mean(nig_cnn_gamma, axis=-1), nig_qpens),
                np.nanmean(
                    _eu_grid_values_from_evidential(
                        nig_cnn_nu, nig_cnn_alpha, nig_cnn_beta
                    ),
                    axis=-1,
                ),
            )
        )
    if ensemble_qpens is not None and ensemble_cnn_mean is not None:
        rows.append(
            (
                "Deep Ensemble",
                _per_seed_time_rmse(
                    _mean_prediction_from_samples(ensemble_cnn_mean), ensemble_qpens
                ),
                np.nanmean(_eu_grid_values_from_mcdo(ensemble_cnn_mean), axis=-1),
            )
        )

    if not rows:
        return

    fig, axes = plt.subplots(
        len(rows), 3, figsize=(15, 4.5 * len(rows)), squeeze=False, sharex=True
    )
    scatter = None

    for row_idx, (method_name, error_rmse, epistemic) in enumerate(rows):
        for var_idx, var_name in enumerate(variable_names):
            ax = axes[row_idx, var_idx]
            time = np.tile(np.arange(error_rmse.shape[1]), error_rmse.shape[0])
            y = error_rmse[:, :, var_idx].reshape(-1)
            color = epistemic[:, :, var_idx].reshape(-1)
            mask = np.isfinite(time) & np.isfinite(y) & np.isfinite(color)
            time = time[mask]
            y = y[mask]
            color = color[mask]

            positive = color[color > 0]
            if positive.size:
                color_values = np.maximum(color, np.min(positive))
                norm = LogNorm(vmin=np.min(positive), vmax=np.max(positive))
            else:
                color_values = color
                norm = None

            scatter = ax.scatter(
                time,
                y,
                c=color_values,
                norm=norm,
                cmap="viridis",
                s=12,
                alpha=0.75,
                edgecolors="none",
            )
            ax.set_title(f"{method_name} - {var_name}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("RMSE vs QPEns")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Model error over time colored by epistemic uncertainty", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.90, 0.97))
    if scatter is not None:
        colorbar_axis = fig.add_axes([0.92, 0.12, 0.02, 0.76])
        fig.colorbar(scatter, cax=colorbar_axis, label="EU")
    save_path = f"{viz_dir}/{save_name}_error_vs_time_colored_by_eu.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Error vs time colored by EU plot saved to: {save_path}")
    plt.close(fig)
