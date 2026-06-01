import os

from dataclasses import dataclass
from typing import Dict, Any
import copy
import pickle
from pathlib import Path
from datetime import datetime

import tqdm

from msw_da_ml.core.msw_model import EnsembleModel, animate_evolution_from_history
from msw_da_ml.core.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.core.observations import ObservationGenerator
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
global_config = settings.global_config
sequence_path = get_output_dir(global_config.training_sequences_out_filename)


@dataclass
class TrainingSequence:
    models: Dict[str, Any]
    histories: Dict[str, list]
    iteration: int = 0


class TrainingSequenceGenerator:
    def __init__(self, num_ensemble_members: int, rngs):
        self.num_ensemble_members = num_ensemble_members
        self.rngs = rngs
        self.observation_generator = ObservationGenerator(rngs)
        self.kf_assimilator = EnsembleKalmanFilter()
        self.qp_assimilator = QPEnsemble()

    def run(
        self,
        num_steps: int,
        generate_evolution_animations: bool = True,
        save_data: bool = True,
        sequence_save_name: str = "cnn_training_sequence",
        include_timestamp_in_name: bool = True,
    ):
        """
        Generates Truth, QPEns and EnKF training sequences.
        This data may be used for verifying or testing different msw model configurations.
        The main utility for the generated data is as training data for the CNN.
        """
        if include_timestamp_in_name:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            sequence_save_name += f"_{timestamp}"

        model_truth = EnsembleModel(
            num_ensemble_members=1, random_generator=self.rngs.truth_rng
        )
        model_ensemble = EnsembleModel(
            num_ensemble_members=self.num_ensemble_members,
            random_generator=self.rngs.ensemble_rng,
        )
        model_truth.initialize()
        model_ensemble.initialize()

        state = TrainingSequence(
            models={
                "truth": model_truth,
                "ensemble_kf": copy.deepcopy(model_ensemble),
                "ensemble_qp": copy.deepcopy(model_ensemble),
            },
            histories={"kf": [], "qp": [], "observation_locations": []},
        )

        for i in tqdm.tqdm(range(num_steps), desc="Pipeline Progress"):
            state.iteration = i
            state = self._forecast_backgrounds(state)
            state = self._generate_observations(state)
            state = self._assimilate(state)

        if generate_evolution_animations:
            self._generate_animations(state, sequence_save_name)

        if save_data:
            self._save_training_sequence(state, sequence_name=sequence_save_name)

        return state

    def _save_training_sequence(
        self, state: TrainingSequence, sequence_name: str
    ):
        save_path = os.path.join(sequence_path, sequence_name)

        if not sequence_name.endswith(".pkl"):
            save_path += ".pkl"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Training sequence saved to: {save_path}")

    @staticmethod
    def load_training_sequence(sequence_name: str) -> TrainingSequence:
        load_path = os.path.join(sequence_path, sequence_name)

        if not sequence_name.endswith(".pkl"):
            load_path += ".pkl"

        print(load_path)
        with open(load_path, "rb") as f:
            state = pickle.load(f)
        print(f"Training sequence loaded from: {load_path}")
        return state

    def _forecast_backgrounds(self, state: TrainingSequence):
        """Step 1: Forecast truth and the QPEns-cycled background."""
        state.models["truth"].propagate()
        state.models["ensemble_qp"].propagate()
        return state

    def _generate_observations(self, state: TrainingSequence):
        """Step 2: Observation generation from truth"""
        truth_state = state.models["truth"].get_state()

        obs_data = self.observation_generator.generate_observations_with_locations(
            truth_state, self.num_ensemble_members
        )
        state.observations = obs_data.observation
        state.observation_locations = obs_data.locations
        state.histories["observation_locations"].append(obs_data.locations)
        return state

    def _assimilate(self, state: TrainingSequence):
        """Step 3: Assimilate ensemble models"""
        # This part may be adjusted. For propagation to the next EnKF state, we take
        # the previous QPEns state. We do this since this data is used for model training.
        # In each CNN adjustment we assume that the previous state is a QPEns adjusted state,
        # since the CNN is supposed to mimic the QPEns behavior.
        ensemble_state_kf = state.models["ensemble_qp"].get_state().copy()
        ensemble_state_qp = state.models["ensemble_qp"].get_state().copy()

        kf_assimilated = self.kf_assimilator.assimilate(
            ensemble=ensemble_state_kf,
            observation=state.observations,
            observation_position=state.observation_locations,
        )
        qp_assimilated = self.qp_assimilator.assimilate(
            ensemble=ensemble_state_qp,
            observation=state.observations,
            observation_position=state.observation_locations,
        )

        state.histories["kf"].append(kf_assimilated)
        state.histories["qp"].append(qp_assimilated)

        state.models["ensemble_kf"].assimilate(kf_assimilated)
        state.models["ensemble_qp"].assimilate(qp_assimilated)

        return state

    def _generate_animations(
        self, state: TrainingSequence, sequence_save_name: str
    ):
        """Generate output animations"""
        animations_dir = get_output_dir(
            global_config.generated_data_animations_out_filename
        )

        kf_animation_path = os.path.join(
            animations_dir, f"{sequence_save_name}_evolution_kf.mp4"
        )
        qp_animation_path = os.path.join(
            animations_dir, f"{sequence_save_name}_evolution_qp.mp4"
        )
        truth_animation_path = os.path.join(
            animations_dir, f"{sequence_save_name}_evolution_truth.mp4"
        )

        animate_evolution_from_history(state.histories["kf"], kf_animation_path)
        animate_evolution_from_history(state.histories["qp"], qp_animation_path)
        state_history_truth = state.models["truth"].get_history()
        animate_evolution_from_history(state_history_truth, truth_animation_path)
        print(
            f"Evolution Animations for EnKF, QPens and Truth saved to: {animations_dir}"
        )
