import os

from dataclasses import dataclass
from typing import Dict, Any
import copy
import pickle
from pathlib import Path
from datetime import datetime

import tqdm

from msw_da_ml.msw.msw_model import EnsembleModel, animate_evolution_from_history
from msw_da_ml.msw.assimilation import EnsembleKalmanFilter, QPEnsemble
from msw_da_ml.msw.observation_generation import ObservationGenerator
from msw_da_ml.settings import load_settings, get_output_dir

settings = load_settings()
global_config = settings.global_config
state_path = get_output_dir(global_config.msw_model_out_filename)


@dataclass
class DataGenerationState:
    models: Dict[str, Any]
    histories: Dict[str, list]
    iteration: int = 0


class DataGenerationPipeline:
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
        pipeline_state_save_name: str = "cnn_training_data",
        include_timestamp_in_name: bool = True
    ):
        """
        Generates Truth, QPEns and EnKF data history.
        This data may be used for verifying or testing different msw model configurations.
        The main utility for the generated data is as training data for the CNN.
        """
        if include_timestamp_in_name:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            pipeline_state_save_name += f"_{timestamp}"
        
        model_truth = EnsembleModel(
            num_ensemble_members=1, random_generator=self.rngs.truth_rng
        )
        model_ensemble = EnsembleModel(
            num_ensemble_members=self.num_ensemble_members,
            random_generator=self.rngs.ensemble_rng,
        )
        model_truth.initialize()
        model_ensemble.initialize()

        state = DataGenerationState(
            models={
                "truth": model_truth,
                "ensemble_kf": copy.deepcopy(model_ensemble),
                "ensemble_qp": copy.deepcopy(model_ensemble),
            },
            histories={"kf": [], "qp": [], "observation_locations": []},
        )

        for i in tqdm.tqdm(range(num_steps), desc="Pipeline Progress"):
            state.iteration = i
            state = self._forecast_truth(state)
            state = self._generate_observations(state)
            state = self._assimilate_and_forecast(state)

        if generate_evolution_animations:
            self._generate_animations(state, pipeline_state_save_name)

        if save_data:
            self._save_pipeline_state(
                state, pipeline_state_name=pipeline_state_save_name
            )

        return state

    def _save_pipeline_state(
        self, state: DataGenerationState, pipeline_state_name: str
    ):
        save_path = os.path.join(state_path, pipeline_state_name)

        if not pipeline_state_name.endswith(".pkl"):
            save_path += ".pkl"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Pipeline state saved to: {save_path}")

    @staticmethod
    def load_pipeline_state(pipeline_state_name: str) -> DataGenerationState:
        load_path = os.path.join(state_path, pipeline_state_name)

        if not pipeline_state_name.endswith(".pkl"):
            load_path += ".pkl"

        print(load_path)
        with open(load_path, "rb") as f:
            state = pickle.load(f)
        print(f"Pipeline state loaded from: {load_path}")
        return state

    def _forecast_truth(self, state: DataGenerationState):
        """Step 1: Truth model step"""
        if state.iteration > 0:
            state.models["truth"].propagate()
        return state

    def _generate_observations(self, state: DataGenerationState):
        """Step 2: Observation generation from truth"""
        truth_state = state.models["truth"].get_state()

        obs_data = self.observation_generator.generate_observations_with_locations(
            truth_state, self.num_ensemble_members
        )
        state.observations = obs_data.observation
        state.observation_locations = obs_data.locations
        state.histories["observation_locations"].append(obs_data.locations)
        return state

    def _assimilate_and_forecast(self, state: DataGenerationState):
        """Step 3: Assimilate and forecast ensemble models"""
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
        state.models["ensemble_kf"].propagate()
        state.models["ensemble_qp"].assimilate(qp_assimilated)
        state.models["ensemble_qp"].propagate()

        return state

    def _generate_animations(self, state: DataGenerationState, pipeline_state_save_name: str):
        """Generate output animations"""
        kf_animation_path = get_output_dir(global_config.generated_data_animations_out_filename)
        qp_animation_path = get_output_dir(global_config.generated_data_animations_out_filename)
        f"{pipeline_state_save_name}_evolution_kf.mp4"
        f"{pipeline_state_save_name}_model_evolution_qp.mp4"
        animate_evolution_from_history(
            state.histories["kf"], str(kf_animation_path)
        )
        animate_evolution_from_history(
            state.histories["qp"], str(qp_animation_path)
        )
        state_history_truth = state.models["truth"].get_history()
        animate_evolution_from_history(state_history_truth)
