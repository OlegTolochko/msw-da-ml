from pathlib import Path
import os

import cyclopts

from msw_da_ml.settings import load_settings, get_output_dir
from msw_da_ml.msw.random_manager import RandomGenerators
from msw_da_ml.msw.msw_data_generation import DataGenerationPipeline
from msw_da_ml.msw_cnn.train_nn import train_nn
from msw_da_ml.conformal_quantile_regression.qr_train_nn import train_quantile_nn
from msw_da_ml.msw_cnn.inference import inference, get_most_recent_model_name
from msw_da_ml.conformal_prediction.cp_data_generation import (
    generate_experiment_data as generate_cp_experiment_data,
)
from msw_da_ml.conformal_quantile_regression.cqr_data_generation import (
    generate_experiment_data_qr as generate_cqr_experiment_data,
)
from msw_da_ml.uq_comparison_pipeline import generate_comparison_analysis
from msw_da_ml.conformal_prediction.conformal_prediction import conformal_prediction
from msw_da_ml.conformal_quantile_regression.cqr_prediction import cqr_prediction

settings = load_settings()
global_config = settings.global_config

rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)

app = cyclopts.App(
    "For further more detailed settings you may look into the config.yaml"
)


@app.command()
def generate_training_data(
    num_ensemble_members: int = 10,
    num_steps: int = 20,
    save_name: str = "training_data",
):
    """Generates Training data for CNN and CQR CNN.

    Parameters
    ----------
    num_ensemble_members: int
        Number of ensemble members.
    num_steps: int
        Number of data generation steps.
    save_time: str
        Save name for generated data.
    """
    pipeline = DataGenerationPipeline(
        num_ensemble_members=num_ensemble_members, rngs=rngs
    )
    pipeline.run(num_steps=num_steps, pipeline_state_save_name=save_name)


@app.command()
def train_cnn_model(
    generated_training_data_name: str,
    model_name: str = "cnn_model",
    include_timestamp_in_name: bool = True,
):
    """Trains a CNN model for data assimilation.

    Parameters
    ----------
    generated_msw_data_name: str
        Name of the generated training data pipeline state file.
    include_timestamp_in_name: bool
        Whether to include timestamp in the model name.
    """
    train_nn(
        generated_training_data_name=generated_training_data_name,
        model_name=model_name,
        include_timestamp_in_name=include_timestamp_in_name,
    )


@app.command()
def train_quantile_regression_model(
    generated_msw_data_name: str,
    quantile_tau: float = 0.9,
    include_timestamp_in_name: bool = True,
):
    """Trains a quantile regression CNN model for uncertainty quantification.

    Parameters
    ----------
    generated_msw_data_name: str
        Name of the generated training data pipeline state file.
    quantile_tau: float
        Quantile level for training (default: 0.9).
    include_timestamp_in_name: bool
        Whether to include timestamp in the model name.
    """
    train_quantile_nn(generated_msw_data_name, quantile_tau, include_timestamp_in_name)


@app.command()
def run_inference(
    num_inference_steps: int = 200,
    load_model_name: str = "",
    compute_qpens: bool = True,
):
    """Runs inference with a trained CNN model and generates comparison visualizations.

    Parameters
    ----------
    num_inference_steps: int
        Number of inference steps to run.
    load_model_name: str
        Name of the model to load (uses latest if empty).
    compute_qpens: bool
        Whether to also compute QPEns predictions for comparison.
    """
    inference(num_inference_steps, load_model_name, compute_qpens)


@app.command()
def generate_conformal_prediction_data(load_model_name: str = ""):
    """Generates experimental data for conformal prediction analysis.

    Parameters
    ----------
    load_model_name: str
        Name of the CNN model to use for experiments.
    """
    generate_cp_experiment_data(load_model_name)


@app.command()
def generate_cqr_data(load_model_name: str = ""):
    """Generates experimental data for conformalized quantile regression analysis.

    Parameters
    ----------
    load_model_name: str
        Name of the quantile regression model to use for experiments.
    """
    generate_cqr_experiment_data(load_model_name)


@app.command()
def run_conformal_prediction(cp_hist_name: str, normalize: bool = False):
    """Runs conformal prediction analysis on experimental data.

    Parameters
    ----------
    cp_hist_name: str
        Name of the conformal prediction experiment data file.
    normalize: bool
        Whether to use normalized conformal prediction.
    """
    conformal_prediction(cp_hist_name, normalize)


@app.command()
def run_cqr_prediction(cqr_hist_name: str):
    """Runs conformalized quantile regression analysis on experimental data.

    Parameters
    ----------
    cqr_hist_name: str
        Name of the CQR experiment data file.
    """
    cqr_prediction(cqr_hist_name)


@app.command()
def compare_uq_methods(
    cp_hist_name: str,
    cqr_hist_name: str,
    mcdo_hist_name: str | None = None,
    nig_hist_name: str | None = None,
    normalize_cp: bool = True,
    include_cnn_std: bool = False,
):
    """Compares conformal prediction vs conformalized quantile regression methods.
    Requires cp and cqr data to be generated by the same config for a fair comparison.

    Parameters
    ----------
    cp_hist_name: str
        Name of the conformal prediction experiment data file.
    cqr_hist_name: str
        Name of the CQR experiment data file.
    normalize_cp: bool
        Whether to also run normalized CP for comparison.
    """
    generate_comparison_analysis(
        cp_hist_name,
        cqr_hist_name,
        mcdo_hist_name,
        nig_hist_name,
        normalize_cp,
        include_cnn_std,
    )


@app.command()
def list_available_models():
    """Lists all available trained models."""
    print("=== Available CNN Models ===")
    cnn_model_dir = get_output_dir(settings.global_config.trained_nn_model_out_filename)
    if cnn_model_dir.exists():
        models = [
            f for f in cnn_model_dir.iterdir() if f.is_file() and f.suffix == ".pth"
        ]
        if models:
            for model in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
                print(f"  {model.name}")
        else:
            print("  No CNN models found")
    else:
        print("  CNN model directory does not exist")

    print("\n=== Available Quantile Regression Models ===")
    qr_model_dir = get_output_dir(
        settings.global_config.trained_quantile_nn_model_out_filename
    )
    if qr_model_dir.exists():
        models = [
            f for f in qr_model_dir.iterdir() if f.is_file() and f.suffix == ".pth"
        ]
        if models:
            for model in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
                print(f"  {model.name}")
        else:
            print("No quantile regression models found")
    else:
        print("Quantile regression model directory does not exist")


@app.command()
def list_available_data():
    """Lists all available training data and experiment files."""
    print("=== Available Training Data ===")
    training_data_dir = get_output_dir(settings.global_config.msw_model_out_filename)
    if training_data_dir.exists():
        data_files = [
            f for f in training_data_dir.iterdir() if f.is_file() and f.suffix == ".pkl"
        ]
        if data_files:
            for data_file in sorted(
                data_files, key=lambda x: x.stat().st_mtime, reverse=True
            ):
                print(f"  {data_file.name}")
        else:
            print("No training data found")
    else:
        print("Training data directory does not exist")

    print("\n=== Available Experiment Data ===")
    exp_data_dir = get_output_dir(
        settings.global_config.experiment_histories_out_filename
    )
    if exp_data_dir.exists():
        exp_files = [
            f for f in exp_data_dir.iterdir() if f.is_file() and f.suffix == ".npz"
        ]
        if exp_files:
            for exp_file in sorted(
                exp_files, key=lambda x: x.stat().st_mtime, reverse=True
            ):
                print(f"  {exp_file.name}")
        else:
            print("No experiment data found")
    else:
        print("Experiment data directory does not exist")

    print("\n=== Available Conformal Quantile Regression Experiment Data ===")
    exp_data_dir = get_output_dir(
        settings.global_config.quantile_experiment_histories_out_filename
    )
    if exp_data_dir.exists():
        exp_files = [
            f for f in exp_data_dir.iterdir() if f.is_file() and f.suffix == ".npz"
        ]
        if exp_files:
            for exp_file in sorted(
                exp_files, key=lambda x: x.stat().st_mtime, reverse=True
            ):
                print(f"  {exp_file.name}")
        else:
            print("No experiment data found")
    else:
        print("Experiment data directory does not exist")


if __name__ == "__main__":
    app()
