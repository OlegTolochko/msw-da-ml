from pathlib import Path
import os

import cyclopts

from msw_da_ml.settings import load_settings, get_output_dir
from msw_da_ml.msw.random_manager import RandomGenerators
from msw_da_ml.msw.msw_data_generation import DataGenerationPipeline
from msw_da_ml.msw_cnn.train_nn import train_nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

settings = load_settings()
global_config = settings.global_config

OUT_DIR = get_output_dir()

rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)

app = cyclopts.App("For further more detailed settings you may look into the config.yaml")

@app.command()
def generate_training_data(
    num_ensemble_members: int = 10,
    num_steps: int = 20,
    save_name: str = "trainig_data"
):
    """Generates Training data for CNN.

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

def train_cnn_model(
    generated_data_name: str
):  
    train_nn(generated_data_name)

def train_quantile_regression_cnn_model():    
    pass

def cp_prediction_pipeline():
    pass

def cqr_prediction_pipeline():
    pass

def compare_cp_cqr():
    pass

if __name__ == "__main__":
    app()