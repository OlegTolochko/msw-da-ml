from collections.abc import Iterator

import numpy as np

from msw_da_ml.core.assimilation import EnsembleKalmanFilter
from msw_da_ml.core.observations import ObservationGenerator
from msw_da_ml.core.random import RandomGenerators
from msw_da_ml.data.evaluation_sequences import EvaluationSequence
from msw_da_ml.settings import load_settings

settings = load_settings()
experiment_config = settings.experiment_config


def as_float32(values: list[np.ndarray]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def iter_cnn_inputs_from_base(
    sequence: EvaluationSequence,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if sequence.cnn_enkf_analysis and sequence.observation_locations:
        yield from zip(sequence.cnn_enkf_analysis, sequence.observation_locations)
        return

    rngs = RandomGenerators.from_seed(sequence.seed)
    obs_generator = ObservationGenerator(rngs)
    enkf = EnsembleKalmanFilter()

    for truth_state, cnn_background in zip(sequence.truth, sequence.cnn_background):
        obs_data = obs_generator.generate_observations_with_locations(
            truth_state, experiment_config.num_ensemble_members
        )
        cnn_enkf_analysis = enkf.assimilate(
            cnn_background, obs_data.observation, obs_data.locations
        )
        yield cnn_enkf_analysis, obs_data.locations


def materialize_cnn_inputs(sequence: EvaluationSequence) -> None:
    if sequence.cnn_enkf_analysis and sequence.observation_locations:
        return

    pairs = list(iter_cnn_inputs_from_base(sequence))
    sequence.cnn_enkf_analysis = [cnn_input.copy() for cnn_input, _ in pairs]
    sequence.observation_locations = [
        observation_locations.copy() for _, observation_locations in pairs
    ]
