import numpy as np
from settings import load_settings

settings = load_settings()
obs_config = settings.observation_generation_config


def generate_observations_from_state_history(state_history, random_generator):
    pass


def generate_observation(
    truth_state: np.ndarray,
    num_ensemble_members: int,
    random_generator: np.random.Generator,
):
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


def generate_radar_masks(
    truth_state: np.ndarray, random_generator: np.random.Generator
):
    num_grid_cells = truth_state.shape[1]

    u_mask = np.ones(num_grid_cells)
    h_mask = np.ones(num_grid_cells)
    r_mask = np.ones(num_grid_cells)

    is_raining = truth_state[2, :, 0] > obs_config.radar_rain_threshold

    clear_sky_observations = random_generator.choice(
        a=[True, False],
        size=num_grid_cells,
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


def assimilate(ensemble, observation, observation_position, assimilation_method):
    observation_position = np.concat(observation_position, axis=0)
    
    num_ensemble_members = ensemble.shape[0]
    u_var = obs_config.u_error_std**2
    h_var = obs_config.h_error_std**2
    r_var = (np.exp(obs_config.r_error_std**2)*np.exp(2*obs_config.r_error_mean + obs_config.r_error_std**2))
    var_flat = np.concat(u_var, h_var, r_var)
    
    ens_mean_difference = np.sqrt(obs_config.cov_inflation/(num_ensemble_members-1))*(ensemble - np.mean(ensemble, axis=2))
    flat_state = np.concat(ens_mean_difference, axis=0) # flattend to compute cov. err for grid point in all domains (u, h, r) to each other grid point
    cov_error = np.dot(flat_state, flat_state.T)
    
    kalman_gain = np.dot(cov_error[:, observation_position], np.linalg.inv(cov_error[observation_position, observation_position] + np.diag(var_flat)))
