from cyclopts import App

from msw_da_ml.msw.random_manager import RandomGenerators
from msw_da_ml.settings import load_settings, get_output_dir
from msw_da_ml.data.training_sequences import TrainingSequenceGenerator
from msw_da_ml.data.evaluation_sequences import generate_evaluation_data


app = App()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=settings.global_config.base_seed)
# Ensure output directory exists
get_output_dir()


@app.command()
def generate_cnn_training_data(
    num_ensemble_members: int = 10,
    num_steps: int = 20,
    sequence_save_name: str = "cnn_training_sequence",
):
    generator = TrainingSequenceGenerator(
        num_ensemble_members=num_ensemble_members, rngs=rngs
    )
    generator.run(num_steps=num_steps, sequence_save_name=sequence_save_name)


@app.command()
def generate_evaluation_sequence_data():
    generate_evaluation_data()


if __name__ == "__main__":
    app()
