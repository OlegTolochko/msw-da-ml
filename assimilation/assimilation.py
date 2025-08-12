from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.settings import load_settings
import cvxopt

settings = load_settings()
assimilation_config = settings.shared_obs_gen_assimilation_config


class BaseAssimilation(ABC):
    @abstractmethod
    def assimilate(self, ensemble, obseravtion, obseravation_mask):
        pass

    def calculate_gaspari_cohn(self, z: np.ndarray):
        """
        Calculates the Gaspari-Cohn function for normalized distances z.
        Effectively ensures that neighbors that are further away are weighted less.
        """
        rho = np.zeros_like(z, dtype=float)

        mask1 = (z >= 0) & (z < 1)
        rho[mask1] = (
            -0.25 * z[mask1] ** 5
            + 0.5 * z[mask1] ** 4
            + (5.0 / 8.0) * z[mask1] ** 3
            - (5.0 / 3.0) * z[mask1] ** 2
            + 1.0
        )

        mask2 = (z >= 1) & (z < 2)
        rho[mask2] = (
            (1.0 / 12.0) * z[mask2] ** 5
            - 0.5 * z[mask2] ** 4
            + (5.0 / 8.0) * z[mask2] ** 3
            + (5.0 / 3.0) * z[mask2] ** 2
            - 5.0 * z[mask2]
            + 4.0
            - (2.0 / 3.0) * (1.0 / z[mask2])
        )

        return rho

    def calculate_localization_matrix(
        self, num_grid_points: int, grid_point_influence: int
    ):
        """
        Calculates the localization matrix.
        Limits neighboring grid cell influence with a max influence distance of grid_point_influence.
        """
        if grid_point_influence == 0:
            return np.identity(num_grid_points * 3)

        indices = np.arange(num_grid_points)
        distances = np.abs(indices - indices[:, np.newaxis])
        periodic_distances = np.minimum(distances, num_grid_points - distances)

        z = periodic_distances / float(grid_point_influence)

        localization_matrix = self.calculate_gaspari_cohn(z)

        expanded_localization_matrix = np.tile(localization_matrix, (3, 3))

        return expanded_localization_matrix

    def calculate_obs_covariance_error(
        self, num_grid_points, observation_position_flat
    ):
        u_var = np.tile(assimilation_config.u_error_std**2, num_grid_points)
        h_var = np.tile(assimilation_config.h_error_std**2, num_grid_points)
        r_var = np.tile(
            np.exp(assimilation_config.r_error_std**2)
            * np.exp(
                2 * assimilation_config.r_error_mean
                + assimilation_config.r_error_std**2
            ),
            num_grid_points,
        )
        var_flat = np.concat([u_var, h_var, r_var])[observation_position_flat]
        return var_flat

    def calculate_background_covariance_error(
        self, ensemble: np.ndarray, num_ensemble_members: int, num_grid_points: int
    ):
        """
        Calculates the background error covariance matrix from ensemble deviations.

        It calculates the uncertainty (variance) at each individual grid points across ensemble members
        and the correlations between different grid points and the variables (u, h, r), with a max
        max influence range of grid_point_influence.
        """
        ens_mean_difference = np.sqrt(
            assimilation_config.cov_inflation / (num_ensemble_members - 1)
        ) * (ensemble - np.mean(ensemble, axis=2, keepdims=True))
        flat_state = np.concat(
            ens_mean_difference, axis=0
        )  # flattend to compute cov. err for grid point in all domains (u, h, r) to each other grid point
        cov_error = np.dot(flat_state, flat_state.T)
        cov_error = cov_error * self.calculate_localization_matrix(
            num_grid_points=num_grid_points,
            grid_point_influence=assimilation_config.grid_point_influence,
        )
        return cov_error


class QPEnsemble(BaseAssimilation):
    def assimilate(self, ensemble, observation, observation_position):
        """
        Calculates an ensemble update, while enforcing physical constraints.
        Solves an optimization problem for each ensemble member indepentently.
        """
        observation_position_flat = np.concat(observation_position, axis=0)

        num_ensemble_members = ensemble.shape[2]
        num_grid_points = ensemble.shape[1]

        var_flat = self.calculate_obs_covariance_error(
            num_grid_points=num_grid_points,
            observation_position_flat=observation_position_flat,
        )

        ens_obs_difference = (
            observation[observation_position] - ensemble[observation_position]
        )

        cov_error = self.calculate_background_covariance_error(
            ensemble=ensemble,
            num_ensemble_members=num_ensemble_members,
            num_grid_points=num_grid_points,
        )
        eigen_vectors, singular_values, eigen_vectors_transposed = np.linalg.svd(
            cov_error, full_matrices=True
        )
        cov_error_sqrt = np.dot(eigen_vectors, np.diag(np.sqrt(singular_values)))
        cov_error_sqrt_obs = cov_error_sqrt[observation_position_flat]

        weighed_uncertainty = np.divide(cov_error_sqrt_obs, var_flat[:, None])

        hessian = np.identity(num_grid_points * 3) + np.dot(
            cov_error_sqrt_obs.T, weighed_uncertainty
        )

        rain_mask = np.arange(num_grid_points * 2, num_grid_points * 3)
        height_mask = np.arange(num_grid_points, num_grid_points * 2)

        mass_conservation_constraint = np.dot(
            np.ones(num_grid_points), np.asmatrix(cov_error_sqrt[height_mask])
        )
        results = Parallel(n_jobs=-1)(
            delayed(self.solve_one_ensemble_member)(
                ensemble[2, :, ens_idx],
                ens_obs_difference[:, ens_idx],
                hessian,
                weighed_uncertainty,
                cov_error_sqrt,
                rain_mask,
                mass_conservation_constraint,
            )
            for ens_idx in range(num_ensemble_members)
        )

        qpens_solution = np.stack(results, axis=1)

        ensemble_update_flat = np.dot(cov_error_sqrt, qpens_solution)
        u_update = ensemble_update_flat[0:num_grid_points]
        h_update = ensemble_update_flat[num_grid_points : 2 * num_grid_points]
        r_update = ensemble_update_flat[2 * num_grid_points : 3 * num_grid_points]
        ensemble_update = np.stack([u_update, h_update, r_update])

        ensemble_updated = ensemble + ensemble_update
        return ensemble_updated

    @staticmethod
    def solve_one_ensemble_member(
        ensemble_rain_slice,
        ens_obs_difference_slice,
        hessian,
        weighed_uncertainty,
        cov_error_sqrt,
        rain_mask,
        mass_conservation_constraint,
    ):
        """
        Solves QP problem for single ensemble member
        """
        observation_update_direction = np.dot(
            -weighed_uncertainty.T, ens_obs_difference_slice
        )
        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(
            cvxopt.matrix(hessian),
            cvxopt.matrix(observation_update_direction),
            cvxopt.matrix(-cov_error_sqrt[rain_mask]),
            cvxopt.matrix(ensemble_rain_slice),
            cvxopt.matrix(mass_conservation_constraint),
            cvxopt.matrix(np.zeros((1, 1))),
        )
        return np.asarray(solution["x"]).reshape(-1)


class EnsembleKalmanFilter(BaseAssimilation):
    def assimilate(self, ensemble, observation, observation_position):
        """
        Updates an ensemble state utilizing the Kalman Filter with an observation and its position,
        not guaranteeing physical consistency.
        It decides based on the covariance of the preliminary predictions how much to trust
        the prediction in comparison to the observations.

        notes:
            matrices states are flattend for an efficient and simplified calculation of the kalman update
        """
        observation_position_flat = np.concat(observation_position, axis=0)

        num_ensemble_members = ensemble.shape[2]
        num_grid_points = ensemble.shape[1]

        var_flat = self.calculate_obs_covariance_error(
            num_grid_points=num_grid_points,
            observation_position_flat=observation_position_flat,
        )
        cov_error = self.calculate_background_covariance_error(
            ensemble=ensemble,
            num_ensemble_members=num_ensemble_members,
            num_grid_points=num_grid_points,
        )

        kalman_gain = np.dot(
            cov_error[:, observation_position_flat],
            np.linalg.inv(
                cov_error[np.ix_(observation_position_flat, observation_position_flat)]
                + np.diag(var_flat)
            ),
        )

        # apply kalman update to our ensemble
        ens_obs_difference = (
            observation[observation_position] - ensemble[observation_position]
        )

        ensemble_update_flat = np.concat(ensemble, axis=0) + np.dot(
            kalman_gain, ens_obs_difference
        )

        u_update = ensemble_update_flat[0:num_grid_points]
        h_update = ensemble_update_flat[num_grid_points : 2 * num_grid_points]
        r_update = ensemble_update_flat[2 * num_grid_points : 3 * num_grid_points]
        ensemble_updated = np.stack([u_update, h_update, r_update])

        return ensemble_updated
