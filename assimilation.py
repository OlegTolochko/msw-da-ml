import numpy as np
from settings import load_settings

settings = load_settings()


def generate_observations_from_state_history(state_history, random_generator):
    pass


def generate_observation(
    truth_state: np.ndarray,
    num_ensemble_members: int,
    random_generator: np.random.Generator,
):
    obs_config = settings.observation_generation_config
    num_grid_cells = truth_state.shape[1]
    ensemble_shape = (num_ensemble_members, num_grid_cells)

    u_error = random_generator.normal(
        loc=obs_config.u_error_mean,
        scale=obs_config.u_error_std,
        size=ensemble_shape,
    )
    h_error = random_generator.normal(
        loc=obs_config.h_error_mean,
        scale=obs_config.h_error_std,
        size=ensemble_shape,
    )
    r_error = random_generator.lognormal(
        mean=obs_config.r_error_mean,
        sigma=obs_config.r_error_std,
        size=ensemble_shape,
    )

    error_ensemble = np.stack([u_error, h_error, r_error], axis=1)
    observation_ensemble = truth_state + error_ensemble
    observation_original = observation_ensemble[0]

    return observation_original, observation_ensemble
