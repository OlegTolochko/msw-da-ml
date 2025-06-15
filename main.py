from typer import Typer
from random_manager import RandomGenerators
from msw_model import ModifiedShallowWaterModel
from settings import load_settings

app = Typer()
settings = load_settings()
rngs = RandomGenerators.from_seed(base_seed=42)


@app.command()
def main():
    pass


methods = ["EnKF", "NN", "QPEns"]


@app.command()
def initialize_msw_model(ensemble_members: int = 10):
    state_truth = ModifiedShallowWaterModel(
        num_ensemble_members=1, random_generator=rngs.truth_rng
    )
    state_ensemble = ModifiedShallowWaterModel(
        num_ensemble_members=ensemble_members, random_generator=rngs.ensemble_rng
    )


@app.command()
def train_nn():
    pass


@app.command()
def assimilate():
    pass
