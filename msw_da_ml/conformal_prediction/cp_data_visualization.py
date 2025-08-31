import os
from typing import List
import datetime

import numpy as np
import matplotlib.pyplot as plt

from msw_da_ml.conformal_prediction.cp_data_generation import ExperimentHistory
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
experiment_config = settings.experiment_config

global_config = settings.global_config

experiments_path = get_output_dir(global_config.experiment_histories_out_filename)


def uhr_rmse_comparison(histories: List[ExperimentHistory]):
    """Create RMSE comparison plots for background and analysis states"""
    num_experiments = len(histories)
    num_steps = len(histories[0].truth)

    variables = ["u", "h", "r"]
    methods = ["enkf", "qpens", "cnn"]

    rmse_background = {method: {var: [] for var in variables} for method in methods}
    rmse_analysis = {method: {var: [] for var in variables} for method in methods}

    for hist in histories:
        truth_array = np.array(hist.truth)

        enkf_bg = np.array(hist.enkf_background)
        qpens_bg = np.array(hist.qpens_background)
        cnn_bg = np.array(hist.cnn_background)

        enkf_an = np.array(hist.enkf_analysis)
        qpens_an = np.array(hist.qpens_analysis)
        cnn_an = np.array(hist.cnn_analysis)

        for t in range(num_steps):
            truth_t = truth_array[t]

            for i, (method_bg, method_name) in enumerate(
                [(enkf_bg, "enkf"), (qpens_bg, "qpens"), (cnn_bg, "cnn")]
            ):
                if t < len(method_bg):
                    bg_t = method_bg[t]
                    bg_mean = np.mean(bg_t, axis=2, keepdims=True)

                    for var_idx, var_name in enumerate(variables):
                        rmse_val = np.sqrt(
                            np.mean((truth_t[var_idx] - bg_mean[var_idx]) ** 2)
                        )
                        if len(rmse_background[method_name][var_name]) <= t:
                            rmse_background[method_name][var_name].append([])
                        rmse_background[method_name][var_name][t].append(rmse_val)

            for i, (method_an, method_name) in enumerate(
                [(enkf_an, "enkf"), (qpens_an, "qpens"), (cnn_an, "cnn")]
            ):
                if t < len(method_an):
                    an_t = method_an[t]
                    an_mean = np.mean(an_t, axis=2, keepdims=True)

                    for var_idx, var_name in enumerate(variables):
                        rmse_val = np.sqrt(
                            np.mean((truth_t[var_idx] - an_mean[var_idx]) ** 2)
                        )
                        if len(rmse_analysis[method_name][var_name]) <= t:
                            rmse_analysis[method_name][var_name].append([])
                        rmse_analysis[method_name][var_name][t].append(rmse_val)

    rmse_bg_mean = {method: {var: [] for var in variables} for method in methods}
    rmse_an_mean = {method: {var: [] for var in variables} for method in methods}

    for method in methods:
        for var in variables:
            for t in range(num_steps):
                if (
                    t < len(rmse_background[method][var])
                    and rmse_background[method][var][t]
                ):
                    rmse_bg_mean[method][var].append(
                        np.mean(rmse_background[method][var][t])
                    )
                if (
                    t < len(rmse_analysis[method][var])
                    and rmse_analysis[method][var][t]
                ):
                    rmse_an_mean[method][var].append(
                        np.mean(rmse_analysis[method][var][t])
                    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("RMSE Comparison: Background vs Analysis", fontsize=16)

    colors = {"enkf": "blue", "qpens": "red", "cnn": "green"}
    labels = {"enkf": "EnKF", "qpens": "QPEns", "cnn": "NN"}

    for var_idx, var in enumerate(variables):
        ax = axes[0, var_idx]

        for method in methods:
            if rmse_bg_mean[method][var]:
                time_steps = range(len(rmse_bg_mean[method][var]))
                ax.plot(
                    time_steps,
                    rmse_bg_mean[method][var],
                    color=colors[method],
                    label=labels[method],
                    linewidth=2,
                )

        ax.set_title(f"RMSE {var}")
        ax.set_ylabel("Background")
        ax.grid(True, alpha=0.3)
        if var_idx == 2:
            ax.legend()

    for var_idx, var in enumerate(variables):
        ax = axes[1, var_idx]

        for method in methods:
            if rmse_an_mean[method][var]:
                time_steps = range(len(rmse_an_mean[method][var]))
                ax.plot(
                    time_steps,
                    rmse_an_mean[method][var],
                    color=colors[method],
                    label=labels[method],
                    linewidth=2,
                )

        ax.set_ylabel("Analysis")
        ax.set_xlabel("Time Steps")
        ax.grid(True, alpha=0.3)

    for var_idx in range(3):
        bg_ax = axes[0, var_idx]
        an_ax = axes[1, var_idx]

        bg_ylim = bg_ax.get_ylim()
        an_ylim = an_ax.get_ylim()

        y_min = min(bg_ylim[0], an_ylim[0])
        y_max = max(bg_ylim[1], an_ylim[1])

        bg_ax.set_ylim(y_min, y_max)
        an_ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    viz_dir = get_output_dir(global_config.visualizations_out_filename)
    plt.savefig(
        f"{viz_dir}/{timestamp}_{experiment_config.base_seed}_{experiment_config.num_seeds}_rmse_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
