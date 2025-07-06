import os

from typer import Typer

from random_manager import RandomGenerators
from settings import load_settings
from data_generation_pipeline import DataGenerationPipeline


app = Typer()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)
os.makedirs(settings.global_config.out_path, exist_ok=True)

@app.command()
def generate_training_data(
    num_ensemble_members: int = 10,
    num_steps: int = 20,
):
    pipeline = DataGenerationPipeline(num_ensemble_members=num_ensemble_members, rngs=rngs)
    result = pipeline.run(num_steps=num_steps)


if __name__ == "__main__":
    app()
