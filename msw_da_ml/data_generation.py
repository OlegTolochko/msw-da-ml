import os

from typer import Typer

from msw_da_ml.msw.random_manager import RandomGenerators
from msw_da_ml.settings import load_settings
from msw_da_ml.msw.msw_data_generation import DataGenerationPipeline
from msw_da_ml.conformal_prediction.cp_data_generation import generate_experiment_data


app = Typer()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)
os.makedirs(settings.global_config.out_path, exist_ok=True)


@app.command()
def generate_training_data(
    num_ensemble_members: int = 10,
    num_steps: int = 20,
):
    pipeline = DataGenerationPipeline(
        num_ensemble_members=num_ensemble_members, rngs=rngs
    )
    result = pipeline.run(num_steps=num_steps)


@app.command()
def generate_history_data():
    generate_experiment_data()


if __name__ == "__main__":
    app()
