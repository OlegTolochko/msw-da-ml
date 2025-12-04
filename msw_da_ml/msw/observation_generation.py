import numpy as np
from dataclasses import dataclass
from typing import Tuple

from msw_da_ml.settings import load_settings

settings = load_settings()
obs_config = settings.shared_obs_gen_assimilation_config


@dataclass
class ObservationData:
    observation: np.ndarray
    locations: np.ndarray


class ObservationGenerator:
    def __init__(self, rngs):
        self.obs_random_generator = rngs.obs_rng
        self.radar_random_generator = rngs.radar_rng

    def generate_observations_with_locations(
        self, truth_state: np.ndarray, num_ensemble_members: int
    ) -> ObservationData:
        observation = generate_observation(
            truth_state, num_ensemble_members, self.obs_random_generator
        )
        locations = generate_radar_masks(truth_state, self.radar_random_generator)
        return ObservationData(observation, locations)


def generate_observation(
    truth_state: np.ndarray,
    num_ensemble_members: int,
    random_generator: np.random.Generator,
):
    """
    Generates an ensemble of observations from a passed state.
    """
    num_grid_points = truth_state.shape[1]
    ensemble_shape = (num_ensemble_members, num_grid_points)
    truth_perturb_shape = num_grid_points

    u_error_truth = random_generator.normal(
        loc=obs_config.u_error_mean,
        scale=obs_config.u_error_std,
        size=truth_perturb_shape,
    )
    h_error_truth = random_generator.normal(
        loc=obs_config.h_error_mean,
        scale=obs_config.h_error_std,
        size=truth_perturb_shape,
    )
    r_error_truth = random_generator.lognormal(
        mean=obs_config.r_error_mean,
        sigma=obs_config.r_error_std,
        size=truth_perturb_shape,
    )
    error_truth = np.stack([u_error_truth, h_error_truth, r_error_truth])[..., None]
    truth_perturb = truth_state + error_truth

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

    error_ensemble = np.transpose(
        np.stack([u_error, h_error, r_error], axis=1), (1, 2, 0)
    )
    observation_ensemble = truth_perturb + error_ensemble

    return observation_ensemble


def generate_radar_masks(
    state_truth: np.ndarray, random_generator: np.random.Generator
):
    """
    Calculates a random mask for where observations are actually observed.
    Its idea is to mimic radar behavior. It observes all 3 variables in rainy areas
    and else only wind with a chance set in the config.
    """
    num_grid_points = state_truth.shape[1]

    u_mask = np.ones(num_grid_points)
    h_mask = np.ones(num_grid_points)
    r_mask = np.ones(num_grid_points)

    is_raining = state_truth[2, :, 0] > obs_config.radar_rain_threshold

    clear_sky_observations = random_generator.choice(
        a=[True, False],
        size=num_grid_points,
        p=[
            obs_config.radar_no_rain_observation_percentage,
            1 - obs_config.radar_no_rain_observation_percentage,
        ],
    )

    u_mask[is_raining] = 0
    h_mask[is_raining] = 0
    r_mask[is_raining] = 0

    u_mask[clear_sky_observations] = 0

    return np.stack([u_mask, h_mask, r_mask]) == 0
