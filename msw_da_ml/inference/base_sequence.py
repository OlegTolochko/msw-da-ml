import copy
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from msw_da_ml.core.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.core.msw_model import EnsembleModel
from msw_da_ml.core.observations import ObservationData, ObservationGenerator
from msw_da_ml.core.random import RandomGenerators
from msw_da_ml.data.evaluation_sequences import EvaluationSequence
from msw_da_ml.settings import load_settings

settings = load_settings()
experiment_config = settings.experiment_config


def as_float32(values: list[np.ndarray]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@dataclass
class ClosedLoopBaseContext:
    model: EnsembleModel
    observation_generator: ObservationGenerator
    enkf: EnsembleKalmanFilter


def closed_loop_context_from_base(sequence: EvaluationSequence) -> ClosedLoopBaseContext:
    rngs = RandomGenerators.from_seed(sequence.seed)

    truth_model = EnsembleModel(num_ensemble_members=1, random_generator=rngs.truth_rng)
    truth_model.initialize()

    ensemble_model = EnsembleModel(
        num_ensemble_members=experiment_config.num_ensemble_members,
        random_generator=rngs.ensemble_rng,
    )
    ensemble_model.initialize()

    qpens_model = copy.deepcopy(ensemble_model)
    qpens = QPEnsemble()
    observation_generator = ObservationGenerator(rngs)

    for _ in range(experiment_config.spinup_cycles):
        truth_model.propagate()
        truth_state = truth_model.get_state()
        obs_data = observation_generator.generate_observations_with_locations(
            truth_state, experiment_config.num_ensemble_members
        )
        qpens_model.propagate()
        qpens_state = qpens_model.get_state()
        qpens_assimilated = qpens.assimilate(
            qpens_state, obs_data.observation, obs_data.locations
        )
        qpens_model.assimilate(qpens_assimilated)

    return ClosedLoopBaseContext(
        model=copy.deepcopy(qpens_model),
        observation_generator=observation_generator,
        enkf=EnsembleKalmanFilter(),
    )


def iter_observations_from_base(
    sequence: EvaluationSequence,
    observation_generator: ObservationGenerator,
) -> Iterator[ObservationData]:
    for cycle_idx, truth_state in enumerate(sequence.truth):
        obs_data = observation_generator.generate_observations_with_locations(
            truth_state, experiment_config.num_ensemble_members
        )
        if sequence.observation_locations:
            expected_locations = sequence.observation_locations[cycle_idx]
            if not np.array_equal(obs_data.locations, expected_locations):
                raise ValueError(
                    "Regenerated observation locations do not match the base "
                    f"sequence at cycle {cycle_idx} for seed {sequence.seed}."
                )
        yield obs_data


def rain_unobserved_channel(
    observation_locations: np.ndarray,
    num_ensemble_members: int,
) -> np.ndarray:
    rain_unobserved_indicator = np.logical_not(observation_locations[2:3])
    return np.tile(
        np.expand_dims(rain_unobserved_indicator, axis=-1),
        (1, 1, num_ensemble_members),
    )


def iter_cnn_inputs_from_base(
    sequence: EvaluationSequence,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    raise RuntimeError(
        "Reusing CNN analysis/background history has been removed for UQ "
        "from-base runs. Use closed_loop_context_from_base() with "
        "iter_observations_from_base() so each UQ method has its own closed-loop "
        "propagation."
    )


def materialize_cnn_inputs(sequence: EvaluationSequence) -> None:
    raise RuntimeError(
        "Materializing CNN inputs from base sequences is disabled because it "
        "reuses CNN history across UQ methods."
    )
