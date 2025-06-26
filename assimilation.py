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

    error_ensemble = np.transpose(
        np.stack([u_error, h_error, r_error], axis=1), (1, 2, 0)
    )
    observation_ensemble = truth_state + error_ensemble

    return observation_ensemble


def generate_radar_masks(
    state_truth: np.ndarray, random_generator: np.random.Generator
):
    num_grid_cells = state_truth.shape[1]

    u_mask = np.ones(num_grid_cells)
    h_mask = np.ones(num_grid_cells)
    r_mask = np.ones(num_grid_cells)

    is_raining = state_truth[2, :, 0] > obs_config.radar_rain_threshold

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


def assimilate(ensemble, observation, observation_position):
    ensemble_updated = calculate_kalman_update(
        ensemble=ensemble,
        observation=observation,
        observation_position=observation_position,
    )
    return ensemble_updated


def calculate_localization_matrix(num_grid_points: int, grid_point_influence: int):
    """calculates the localistion matrix to limit the the influence of grid cells to only a radius of grid_point_influence in the covariance matrix"""
    gasperi_cohn_function_left = (
        lambda z: -0.25 * np.power(z, 5)
        + 0.5 * np.power(z, 4)
        + (5.0 / 8.0) * np.power(z, 3)
        - (5.0 / 3.0) * np.power(z, 2)
        + 1.0
    )

    gasperi_cohn_function_right = (
        lambda z, b, l: (1.0 / 12.0) * np.power(z, 5)
        - 0.5 * np.power(z, 4)
        + (5.0 / 8.0) * np.power(z, 3)
        + (5.0 / 3.0) * np.power(z, 2)
        - 5.0 * z
        + 4.0
        - (2.0 / 3.0) * (b / l)
    )

    localization_matrix = np.zeros((num_grid_points, num_grid_points))
    np.fill_diagonal(localization_matrix, 1)

    for point in np.arrange(-grid_point_influence, grid_point_influence):
        point_normalized = point / grid_point_influence
        if point < 0:
            gasp_point = gasperi_cohn_function_left(point_normalized)
        if point > 0:
            gasp_point = gasperi_cohn_function_right(
                gasp_point, grid_point_influence, point
            )

        np.fill_diagonal(localization_matrix[:, point:], gasp_point)
        np.fill_diagonal(
            localization_matrix[num_grid_points - point :, :point], gasp_point
        )

    # expand to all 3 water model domains
    expanded_localization_matrix = np.tile(localization_matrix, (3, 3))
    return expanded_localization_matrix


def calculate_kalman_update(ensemble, observation, observation_position):
    """
    calculates the EnKF update

    notes:
        matrices states are flattend for an efficient and simplified calculation of the kalman update
    """
    observation_position_flat = np.concat(observation_position, axis=0)

    num_ensemble_members = ensemble.shape[0]
    num_grid_cells = ensemble.shape[1]

    u_var = np.tile(obs_config.u_error_std**2, num_grid_cells)
    h_var = np.tile(obs_config.h_error_std**2, num_grid_cells)
    r_var = np.tile(
        np.exp(obs_config.r_error_std**2)
        * np.exp(2 * obs_config.r_error_mean + obs_config.r_error_std**2),
        num_grid_cells,
    )
    var_flat = np.concat([u_var, h_var, r_var])[observation_position_flat]

    ens_mean_difference = np.sqrt(
        obs_config.cov_inflation / (num_ensemble_members - 1)
    ) * (ensemble - np.mean(ensemble, axis=2, keepdims=True))
    flat_state = np.concat(
        ens_mean_difference, axis=0
    )  # flattend to compute cov. err for grid point in all domains (u, h, r) to each other grid point
    cov_error = np.dot(flat_state, flat_state.T)

    kalman_gain = np.dot(
        cov_error[:, observation_position_flat],
        np.linalg.inv(
            cov_error[observation_position_flat, observation_position_flat]
            + np.diag(var_flat)
        ),
    )

    # apply kalman update to our ensemble
    kalman_update = observation[observation_position] - ensemble[observation_position]

    ensmeble_update_flat = np.concat(ensemble, axis=0) + np.dot(
        kalman_gain, kalman_update
    )

    u_update = ensmeble_update_flat[0:num_grid_cells]
    h_update = ensmeble_update_flat[num_grid_cells : 2 * num_grid_cells]
    r_update = ensmeble_update_flat[2 * num_grid_cells : 3 * num_grid_cells]
    ensemble_update = np.stack([u_update, h_update, r_update])

    return ensemble_update
