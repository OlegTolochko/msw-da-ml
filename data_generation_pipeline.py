import os

from dataclasses import dataclass
from typing import Dict, Any
import copy
import pickle
from pathlib import Path

import tqdm

from msw_model import ModifiedShallowWaterModel, animate_evolution_from_history
from assimilation import (
    generate_observation,
    generate_radar_masks,
    kf_assimilate,
    qpens_assimilate,
)
from settings import load_settings

settings = load_settings()
global_config = settings.global_config
state_path = f"{global_config.out_path}{global_config.msw_model_out_filename}"
os.makedirs(state_path, exist_ok=True)


@dataclass
class DataGenerationState:
    models: Dict[str, Any]
    histories: Dict[str, list]
    iteration: int = 0


class DataGenerationPipeline:
    def __init__(self, num_ensemble_members: int, rngs):
        self.num_ensemble_members = num_ensemble_members
        self.rngs = rngs

    def run(
        self,
        num_steps: int,
        generate_evolution_animations: bool = True,
        save_data: bool = True,
        pipeline_state_save_name: str = "pipeline_state",
    ):
        model_truth = ModifiedShallowWaterModel(
            num_ensemble_members=1, random_generator=self.rngs.truth_rng
        )
        model_ensemble = ModifiedShallowWaterModel(
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
            self._generate_animations(state)

        if save_data:
            self._save_pipeline_state(
                state, pipeline_state_name=pipeline_state_save_name
            )

        return state

    def _save_pipeline_state(
        self, state: DataGenerationState, pipeline_state_name: str
    ):
        save_path = f"{state_path}{pipeline_state_name}"

        if not pipeline_state_name.endswith(".pkl"):
            save_path += ".pkl"

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Pipeline state saved to: {save_path}")

    @staticmethod
    def load_pipeline_state(pipeline_state_name: str) -> DataGenerationState:
        load_path = f"{state_path}{pipeline_state_name}"

        if not pipeline_state_name.endswith(".pkl"):
            load_path += ".pkl"

        with open(load_path, "rb") as f:
            state = pickle.load(f)
        print(f"Pipeline state loaded from: {load_path}")
        return state

    def _forecast_truth(self, state: DataGenerationState):
        """Step 1: Truth model step"""
        if state.iteration > 0:
            state.models["truth"].apply_nsub_steps()
        return state

    def _generate_observations(self, state: DataGenerationState):
        """Step 2: Observation generation from truth"""
        truth_state = state.models["truth"].get_current_state()

        state.observations = generate_observation(
            truth_state=truth_state,
            num_ensemble_members=self.num_ensemble_members,
            random_generator=self.rngs.obs_rng,
        )
        observation_locations = generate_radar_masks(
            state_truth=truth_state, random_generator=self.rngs.radar_rng
        )
        state.observation_locations = observation_locations
        state.histories["observation_locations"].append(observation_locations)
        return state

    def _assimilate_and_forecast(self, state: DataGenerationState):
        """Step 3: Assimilate and forecast ensemble models"""
        ensemble_state_kf = state.models["ensemble_kf"].get_current_state()
        ensemble_state_qp = state.models["ensemble_qp"].get_current_state()

        kf_assimilated = kf_assimilate(
            ensemble=ensemble_state_kf,
            observation=state.observations,
            observation_position=state.observation_locations,
        )
        qp_assimilated = qpens_assimilate(
            ensemble=ensemble_state_qp,
            observation=state.observations,
            observation_position=state.observation_locations,
        )

        state.histories["kf"].append(kf_assimilated)
        state.histories["qp"].append(qp_assimilated)

        state.models["ensemble_kf"].apply_nsub_steps(kf_assimilated)
        state.models["ensemble_qp"].apply_nsub_steps(qp_assimilated)

        return state

    def _generate_animations(self, state: DataGenerationState):
        """Generate output animations"""
        animate_evolution_from_history(
            state.histories["kf"], "./out/model_evolution_kf.mp4"
        )
        animate_evolution_from_history(
            state.histories["qp"], "./out/model_evolution_qp.mp4"
        )
        state_history_truth = state.models["truth"].get_nsub_state_history()
        animate_evolution_from_history(state_history_truth)
