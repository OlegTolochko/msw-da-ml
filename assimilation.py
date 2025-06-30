import numpy as np
from settings import load_settings
import cvxopt

settings = load_settings()
obs_config = settings.observation_generation_config


def generate_observation(
    truth_state: np.ndarray,
    num_ensemble_members: int,
    random_generator: np.random.Generator,
):
    num_grid_points = truth_state.shape[1]
    ensemble_shape = (num_ensemble_members, num_grid_points)

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


def kf_assimilate(ensemble, observation, observation_position):
    ensemble_updated = calculate_kalman_update(
        ensemble=ensemble,
        observation=observation,
        observation_position=observation_position,
    )
    return ensemble_updated


def qpens_assimilate(ensemble, observation, observation_position):
    observation_position_flat = np.concat(observation_position, axis=0)

    num_ensemble_members = ensemble.shape[0]
    num_grid_points = ensemble.shape[1]

    var_flat = calculate_obs_covariance_error(
        num_ensemble_members=num_ensemble_members,
        observation_position_flat=observation_position_flat,
    )

    ens_obs_difference = (
        observation[observation_position] - ensemble[observation_position]
    )

    cov_error = calculate_background_covariance_error(
        ensemble=ensemble,
        num_ensemble_members=num_ensemble_members,
        num_grid_points=num_grid_points,
    )
    eigen_vectors, singular_values, eigen_vectors_transposed = np.linalg.svd(
        cov_error, full_matrices=True
    )
    cov_error_sqrt = np.dot(eigen_vectors, np.sqrt(singular_values))
    cov_error_sqrt_obs = cov_error_sqrt[observation_position]

    weighed_uncertainty = np.divide(
        cov_error_sqrt_obs, observation_position_flat[:, None]
    )

    hessian = np.identity(num_grid_points, 3) + np.dot(
        cov_error_sqrt_obs.T, weighed_uncertainty
    )

    rain_mask = np.arange(num_grid_points * 2, num_grid_points * 3)
    height_mask = np.arange(num_grid_points, num_grid_points * 2)

    mass_conservation_constraint = np.dot(
        np.ones(num_grid_points), np.asmatrix(eigen_vectors[height_mask])
    )
    qpens_solution = np.zeros((num_grid_points * 3, num_ensemble_members))
    for ens_idx in range(num_ensemble_members):
        observation_update_direction = np.dot(
            -weighed_uncertainty.T, ens_obs_difference[:, ens_idx]
        )
        qpens_solution = cvxopt.solvers.qp(
            cvxopt.matrix(hessian),
            cvxopt.matrix(observation_update_direction),
            cvxopt.matrix(-eigen_vectors[rain_mask]),
            cvxopt.matrix(ensemble[rain_mask, ens_idx]),
            cvxopt.matrix(mass_conservation_constraint),
            cvxopt.matrix(np.zeros((1, 1))),
        )

    ensemble_updated += np.dot(eigen_vectors, qpens_solution)
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

    for point in np.arange(-grid_point_influence, grid_point_influence):
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
    num_grid_points = ensemble.shape[1]

    var_flat = calculate_obs_covariance_error(
        num_ensemble_members=num_ensemble_members,
        observation_position_flat=observation_position_flat,
    )
    cov_error = calculate_background_covariance_error(
        ensemble=ensemble,
        num_ensemble_members=num_ensemble_members,
        num_grid_points=num_grid_points,
    )

    kalman_gain = np.dot(
        cov_error[:, observation_position_flat],
        np.linalg.inv(
            cov_error[observation_position_flat, observation_position_flat]
            + np.diag(var_flat)
        ),
    )

    # apply kalman update to our ensemble
    ens_obs_difference = (
        observation[observation_position] - ensemble[observation_position]
    )

    ensmeble_update_flat = np.concat(ensemble, axis=0) + np.dot(
        kalman_gain, ens_obs_difference
    )

    u_update = ensmeble_update_flat[0:num_grid_points]
    h_update = ensmeble_update_flat[num_grid_points : 2 * num_grid_points]
    r_update = ensmeble_update_flat[2 * num_grid_points : 3 * num_grid_points]
    ensemble_updated = np.stack([u_update, h_update, r_update])

    return ensemble_updated


def calculate_obs_covariance_error(num_grid_points, observation_position_flat):
    u_var = np.tile(obs_config.u_error_std**2, num_grid_points)
    h_var = np.tile(obs_config.h_error_std**2, num_grid_points)
    r_var = np.tile(
        np.exp(obs_config.r_error_std**2)
        * np.exp(2 * obs_config.r_error_mean + obs_config.r_error_std**2),
        num_grid_points,
    )
    var_flat = np.concat([u_var, h_var, r_var])[observation_position_flat]
    return var_flat


def calculate_background_covariance_error(
    ensemble, num_ensemble_members, num_grid_points
):
    ens_mean_difference = np.sqrt(
        obs_config.cov_inflation / (num_ensemble_members - 1)
    ) * (ensemble - np.mean(ensemble, axis=2, keepdims=True))
    flat_state = np.concat(
        ens_mean_difference, axis=0
    )  # flattend to compute cov. err for grid point in all domains (u, h, r) to each other grid point
    cov_error = np.dot(flat_state, flat_state.T)
    cov_error = cov_error * calculate_localization_matrix(
        num_grid_points=num_grid_points,
        grid_point_influence=obs_config.grid_point_influence,
    )
    return cov_error
