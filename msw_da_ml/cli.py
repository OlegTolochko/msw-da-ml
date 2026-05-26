import cyclopts

from msw_da_ml.settings import (
    get_evaluation_sequence_dir,
    get_model_artifact_dir,
    get_output_dir,
    load_settings,
)
from msw_da_ml.core.random import RandomGenerators
from msw_da_ml.data.training_sequences import TrainingSequenceGenerator
from msw_da_ml.training.train_cnn import train_nn
from msw_da_ml.training.train_mcdo import train_mcdo_nn
from msw_da_ml.training.train_nig import train_nig_nn
from msw_da_ml.training.train_cqr import train_quantile_nn
from msw_da_ml.inference.cnn_sequence import inference
from msw_da_ml.data.evaluation_sequences import (
    generate_evaluation_data as generate_cp_evaluation_data,
)
from msw_da_ml.inference.mcdo_sequence import (
    generate_mcdo_evaluation_data,
    generate_mcdo_evaluation_data_from_base,
)
from msw_da_ml.inference.nig_sequence import (
    generate_nig_evaluation_data,
    generate_nig_evaluation_data_from_base,
)
from msw_da_ml.inference.cqr_sequence import (
    generate_cqr_evaluation_data,
    generate_cqr_evaluation_data_from_base,
)
from msw_da_ml.uncertainty.uq_comparison import generate_comparison_analysis
from msw_da_ml.uncertainty.split_cp import conformal_prediction
from msw_da_ml.uncertainty.cqr import cqr_prediction

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
    save_name: str = "cnn_training_sequence",
):
    """Generates training sequence data for CNN and CQR CNN.

    Parameters
    ----------
    num_ensemble_members: int
        Number of ensemble members.
    num_steps: int
        Number of data generation steps.
    save_name: str
        Save name for the generated sequence.
    """
    generator = TrainingSequenceGenerator(
        num_ensemble_members=num_ensemble_members, rngs=rngs
    )
    generator.run(num_steps=num_steps, sequence_save_name=save_name)


@app.command()
def train_cnn_model(
    training_sequence_name: str,
    model_name: str = "cnn_model",
    include_timestamp_in_name: bool = True,
):
    """Trains a CNN model for data assimilation.

    Parameters
    ----------
    training_sequence_name: str
        Name of the generated training sequence file.
    include_timestamp_in_name: bool
        Whether to include timestamp in the model name.
    """
    train_nn(
        training_sequence_name=training_sequence_name,
        model_name=model_name,
        include_timestamp_in_name=include_timestamp_in_name,
    )


@app.command()
def train_quantile_regression_model(
    training_sequence_name: str,
    quantile_tau: float = 0.9,
    include_timestamp_in_name: bool = True,
):
    """Trains a quantile regression CNN model for uncertainty quantification.

    Parameters
    ----------
    training_sequence_name: str
        Name of the generated training sequence file.
    quantile_tau: float
        Quantile level for training (default: 0.9).
    include_timestamp_in_name: bool
        Whether to include timestamp in the model name.
    """
    train_quantile_nn(
        training_sequence_name=training_sequence_name,
        quantile_tau=quantile_tau,
        include_timestamp_in_name=include_timestamp_in_name,
    )


@app.command()
def train_mcdo_model(
    training_sequence_name: str,
    model_name: str = "mcdo_cnn_model",
    include_timestamp_in_name: bool = True,
):
    """Trains an MCDO CNN model on a generated training sequence."""
    train_mcdo_nn(
        training_sequence_name=training_sequence_name,
        model_name=model_name,
        include_timestamp_in_name=include_timestamp_in_name,
    )


@app.command()
def train_nig_model(
    training_sequence_name: str,
    model_name: str = "nig_cnn_model",
    include_timestamp_in_name: bool = True,
):
    """Trains a NIG evidential CNN model on a generated training sequence."""
    train_nig_nn(
        training_sequence_name=training_sequence_name,
        model_name=model_name,
        include_timestamp_in_name=include_timestamp_in_name,
    )


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
    """Generates evaluation sequence data for conformal prediction analysis.

    Parameters
    ----------
    load_model_name: str
        Name of the CNN model to use for experiments.
    """
    generate_cp_evaluation_data(load_model_name)


@app.command()
def generate_cqr_data(load_model_name: str = ""):
    """Generates evaluation sequence data for conformalized quantile regression analysis.

    Parameters
    ----------
    load_model_name: str
        Name of the quantile regression model to use for experiments.
    """
    generate_cqr_evaluation_data(load_model_name)


@app.command()
def generate_cqr_data_from_base(base_sequence_name: str, load_model_name: str = ""):
    """Generates CQR predictions from a saved CNN base evaluation sequence."""
    generate_cqr_evaluation_data_from_base(base_sequence_name, load_model_name)


@app.command()
def generate_mcdo_data(load_model_name: str = ""):
    """Generates MCDO evaluation sequence data."""
    generate_mcdo_evaluation_data(load_model_name)


@app.command()
def generate_mcdo_data_from_base(base_sequence_name: str, load_model_name: str = ""):
    """Generates MCDO predictions from a saved CNN base evaluation sequence."""
    generate_mcdo_evaluation_data_from_base(base_sequence_name, load_model_name)


@app.command()
def generate_nig_data(load_model_name: str = ""):
    """Generates NIG evaluation sequence data."""
    generate_nig_evaluation_data(load_model_name)


@app.command()
def generate_nig_data_from_base(base_sequence_name: str, load_model_name: str = ""):
    """Generates NIG predictions from a saved CNN base evaluation sequence."""
    generate_nig_evaluation_data_from_base(base_sequence_name, load_model_name)


@app.command()
def run_conformal_prediction(evaluation_sequence_name: str, normalize: bool = False):
    """Runs conformal prediction analysis on evaluation sequence data.

    Parameters
    ----------
    evaluation_sequence_name: str
        Name of the conformal prediction evaluation sequence file.
    normalize: bool
        Whether to use normalized conformal prediction.
    """
    conformal_prediction(evaluation_sequence_name, normalize)


@app.command()
def run_cqr_prediction(cqr_evaluation_sequence_name: str):
    """Runs conformalized quantile regression analysis on evaluation sequence data.

    Parameters
    ----------
    cqr_evaluation_sequence_name: str
        Name of the CQR evaluation sequence file.
    """
    cqr_prediction(cqr_evaluation_sequence_name)


@app.command()
def compare_uq_methods(
    cp_sequence_name: str,
    cqr_sequence_name: str,
    mcdo_sequence_name: str | None = None,
    nig_sequence_name: str | None = None,
    normalize_cp: bool = True,
    include_cnn_std: bool = False,
    include_rf: bool = False,
    ens_mean: bool = True,
):
    """Compares conformal prediction vs conformalized quantile regression methods.
    Requires CP and CQR sequences to be generated by the same config for a fair comparison.

    Parameters
    ----------
    cp_sequence_name: str
        Name of the conformal prediction evaluation sequence file.
    cqr_sequence_name: str
        Name of the CQR evaluation sequence file.
    normalize_cp: bool
        Whether to also run normalized CP for comparison.
    """
    generate_comparison_analysis(
        cp_sequence_name=cp_sequence_name,
        cqr_sequence_name=cqr_sequence_name,
        mcdo_sequence_name=mcdo_sequence_name or "",
        nig_sequence_name=nig_sequence_name or "",
        normalize_cp=normalize_cp,
        include_cnn_std=include_cnn_std,
        include_rf=include_rf,
        ens_mean=ens_mean,
    )


@app.command()
def list_available_models():
    """Lists all available trained models."""
    for method in ("cnn", "cqr", "mcdo", "nig"):
        print(f"=== Available {method.upper()} Models ===")
        model_dir = get_model_artifact_dir(method)
        models = [
            f for f in model_dir.iterdir() if f.is_file() and f.suffix == ".pth"
        ]
        if models:
            for model in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
                norm_path = model.with_name(f"norm_{model.stem}.pt")
                norm_status = " + norm" if norm_path.exists() else ""
                print(f"  {model.name}{norm_status}")
        else:
            print(f"  No {method.upper()} models found")
        print()


@app.command()
def list_available_data():
    """Lists all available training and evaluation sequence files."""
    print("=== Available Training Sequences ===")
    training_data_dir = get_output_dir(
        settings.global_config.training_sequences_out_filename
    )
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
            print("No training sequences found")
    else:
        print("Training sequence directory does not exist")

    for method in ("cnn", "cqr", "mcdo", "nig"):
        print(f"\n=== Available {method.upper()} Evaluation Sequences ===")
        exp_data_dir = get_evaluation_sequence_dir(method)
        exp_files = [
            f for f in exp_data_dir.iterdir() if f.is_file() and f.suffix == ".npz"
        ]
        if exp_files:
            for exp_file in sorted(
                exp_files, key=lambda x: x.stat().st_mtime, reverse=True
            ):
                print(f"  {exp_file.name}")
        else:
            print(f"No {method.upper()} evaluation sequences found")


if __name__ == "__main__":
    app()
