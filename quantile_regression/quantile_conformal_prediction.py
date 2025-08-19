import numpy as np
from sklearn.model_selection import train_test_split
from typer import Typer
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantile_experiment_data_generation import load_histories
from core.settings import load_settings

app = Typer()

settings = load_settings()
config = settings.conformal_prediction_config
global_config = settings.global_config

viz_dir = f"{global_config.out_path}{global_config.visualizations_out_filename}"
os.makedirs(viz_dir, exist_ok=True)


@app.command()
def quantile_conformal_prediction(hist_name: str, normalize: bool = False):
    """
    Runs conformal prediction pipeline for quantile regression models.
    """
    histories = load_histories(hist_name)
    truth_hist = np.asarray([history.truth for history in histories])
    qpens_hist = np.asarray([history.qpens_analysis for history in histories])
    cnn_lower_hist = np.asarray([history.cnn_analysis_lower_quantiles for history in histories])
    cnn_upper_hist = np.asarray([history.cnn_analysis_upper_quantiles for history in histories])

    # Split data for calibration and testing
    truth_calib, truth_test, qpens_calib, qpens_test, cnn_lower_calib, cnn_lower_test, cnn_upper_calib, cnn_upper_test = (
        train_test_split(
            truth_hist,
            qpens_hist,
            cnn_lower_hist,
            cnn_upper_hist,
            test_size=1 - config.calibration_split_ratio,
            random_state=config.calibration_split_seed,
        )
    )

    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]

    lower_adjustment, upper_adjustment = calibrate_quantile_intervals(
        truth_calib=qpens_calib,  # Use QPEns as ground truth for CNN
        cnn_lower_calib=cnn_lower_calib,
        cnn_upper_calib=cnn_upper_calib,
        normalize=normalize
    )

    cnn_lower_test_adjusted, cnn_upper_test_adjusted = apply_quantile_adjustments(
        cnn_lower_test, cnn_upper_test, lower_adjustment, upper_adjustment
    )

    # Check coverage
    coverage = check_quantile_coverage(qpens_test, cnn_lower_test_adjusted, cnn_upper_test_adjusted)
    
    print(f"Target coverage: {config.calibration_quantile:.0%}")
    print(f"Actual coverage: {np.mean(coverage):.2%}")
    
    # Visualizations
    if normalize:
        hist_name += "_normalized"

    visualize_quantile_coverage(coverage, hist_name)
    visualize_quantile_intervals(cnn_lower_test_adjusted, cnn_upper_test_adjusted, hist_name)
    visualize_quantile_gridpoints(
        cnn_lower_test_adjusted, cnn_upper_test_adjusted, truth_test, qpens_test, hist_name
    )


def calculate_empirical_quantile(scores: np.ndarray, alpha: float):
    num_seeds, *mid, num_grid_points = scores.shape
    s = scores.reshape(num_seeds * num_grid_points, *mid)
    m = s.shape[0]
    k = int(np.ceil((1.0 - alpha) * (m + 1))) - 1
    k = np.clip(k, 0, m - 1)
    s_part = np.partition(s, k, axis=0)
    return s_part[k]


def calibrate_quantile_intervals_symmetric(truth_calib, cnn_lower_calib, cnn_upper_calib):
    """
    Calibrate quantile intervals using conformal prediction.
    
    Returns:
        Tuple of adjustment factors for lower and upper quantiles
    """

    truth_mean = np.mean(truth_calib, axis=-1)
    cnn_lower_mean = np.mean(cnn_lower_calib, axis=-1)
    cnn_upper_mean = np.mean(cnn_upper_calib, axis=-1)

    E = np.maximum(cnn_lower_mean - truth_mean, truth_mean - cnn_upper_mean)

    plot_quantile_non_conformity_scores()
    
    pass


def apply_quantile_adjustments(cnn_lower, cnn_upper, lower_adjustment, upper_adjustment):
    """
    Apply conformal adjustments to quantile predictions.
    """
    lower_adj_expanded = np.expand_dims(lower_adjustment, (0, -1))
    upper_adj_expanded = np.expand_dims(upper_adjustment, (0, -1))
    
    lower_adj_tiled = np.tile(lower_adj_expanded, (cnn_lower.shape[0], 1, 1, cnn_lower.shape[-2], 1))
    upper_adj_tiled = np.tile(upper_adj_expanded, (cnn_upper.shape[0], 1, 1, cnn_upper.shape[-2], 1))
    
    adjusted_lower = cnn_lower - lower_adj_tiled
    adjusted_upper = cnn_upper + upper_adj_tiled
    
    return adjusted_lower, adjusted_upper


def check_quantile_coverage(test_set, lower_quantiles, upper_quantiles):
    """
    Check coverage of quantile intervals.
    """
    test_mean = np.mean(test_set, axis=-1)
    lower_mean = np.mean(lower_quantiles, axis=-1)
    upper_mean = np.mean(upper_quantiles, axis=-1)
    
    coverage = (test_mean >= lower_mean) & (test_mean <= upper_mean)
    return coverage


def visualize_quantile_coverage(coverage, hist_name):
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
        
        ax.plot(timesteps, coverage_mean[:, i], 'b-', linewidth=2, label='Actual Coverage')
        ax.fill_between(
            timesteps,
            coverage_mean[:, i] - coverage_std[:, i],
            coverage_mean[:, i] + coverage_std[:, i],
            alpha=0.3, color='blue', label='±1 Std Dev'
        )
        
        ax.axhline(y=config.calibration_quantile, color='r', linestyle='--', 
                  linewidth=2, label=f'Target ({config.calibration_quantile:.0%})')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Coverage')
        ax.set_title(f'{var_name} Coverage over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    base_name = hist_name.replace('.npz', '')
    save_path = f"{viz_dir}{base_name}_quantile_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Quantile coverage visualization saved to: {save_path}")


def visualize_quantile_intervals(lower_quantiles, upper_quantiles, hist_name):
    """
    Visualize quantile interval widths over time.
    """
    lower_mean = np.mean(lower_quantiles, axis=(0, -1, -2))  # Average over seeds, grid, ensemble
    upper_mean = np.mean(upper_quantiles, axis=(0, -1, -2))
    interval_widths = upper_mean - lower_mean  # Shape: (timesteps, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]
    
    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        timesteps = range(len(interval_widths))
        ax.plot(timesteps, interval_widths[:, i], 'b-', linewidth=2, label='Interval Width')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Interval Width')
        ax.set_title(f'{var_name} Quantile Interval Width')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    base_name = hist_name.replace('.npz', '')
    save_path = f"{viz_dir}{base_name}_quantile_intervals.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Quantile interval visualization saved to: {save_path}")


def visualize_quantile_gridpoints(
    lower_quantiles, upper_quantiles, truth, qpens, hist_name,
    random_seed=0, timestep=50
):
    """
    Visualize quantile intervals at specific seed and timestep.
    """
    if random_seed >= len(truth) or timestep >= len(truth[0]):
        print(f"Warning: Seed {random_seed} or timestep {timestep} out of range")
        return
    
    truth_mean = np.mean(truth[random_seed, timestep], axis=-1)
    qpens_mean = np.mean(qpens[random_seed, timestep], axis=-1)
    lower_mean = np.mean(lower_quantiles[random_seed, timestep], axis=-1)
    upper_mean = np.mean(upper_quantiles[random_seed, timestep], axis=-1)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]
    
    for i, (ax, var_name) in enumerate(zip(axes, variable_names)):
        gridpoints = range(len(truth_mean[i]))
        
        ax.fill_between(gridpoints, lower_mean[i], upper_mean[i],
                       alpha=0.3, color='lightblue', label='Quantile Interval')
        
        ax.plot(gridpoints, truth_mean[i], 'go-', markersize=3, 
               label='Truth', alpha=0.8)
        ax.plot(gridpoints, qpens_mean[i], 'bs-', markersize=3, 
               label='QPEns', alpha=0.8)
        ax.plot(gridpoints, (lower_mean[i] + upper_mean[i])/2, 'r^-', 
               markersize=3, label='Quantile Midpoint', alpha=0.8)
        
        ax.set_xlabel('Grid Point')
        ax.set_ylabel(var_name)
        ax.set_title(f'{var_name} at Seed {random_seed}, Timestep {timestep}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    base_name = hist_name.replace('.npz', '')
    save_path = f"{viz_dir}{base_name}_quantile_gridpoints_seed{random_seed}_t{timestep}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Quantile gridpoint visualization saved to: {save_path}")


def plot_quantile_non_conformity_scores(lower_violations, upper_violations, 
                                       random_seed=0, timestep=50):
    """
    Plot non-conformity scores for quantile regression.
    """
    if random_seed >= len(lower_violations) or timestep >= len(lower_violations[0]):
        return
    
    lower_scores = lower_violations[random_seed, timestep]
    upper_scores = upper_violations[random_seed, timestep]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    variable_names = ["Velocity (u)", "Height (h)", "Rain (r)"]
    
    for i, var_name in enumerate(variable_names):
        axes[i, 0].hist(lower_scores[i], bins=30, alpha=0.7, color='blue',
                       edgecolor='black', label=f'{var_name} Lower Violations')
        axes[i, 0].set_xlabel('Lower Violation Score')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].set_title(f'{var_name} Lower Violations')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].hist(upper_scores[i], bins=30, alpha=0.7, color='red',
                       edgecolor='black', label=f'{var_name} Upper Violations')
        axes[i, 1].set_xlabel('Upper Violation Score')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].set_title(f'{var_name} Upper Violations')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"{viz_dir}quantile_non_conformity_seed{random_seed}_t{timestep}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Non-conformity scores saved to: {save_path}")


if __name__ == "__main__":
    app()
