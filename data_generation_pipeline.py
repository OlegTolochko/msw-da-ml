from dataclasses import dataclass
from typing import Dict, Any
import copy

import tqdm

from msw_model import ModifiedShallowWaterModel
from assimilation import (
    generate_observation,
    generate_radar_masks,
    kf_assimilate,
    qpens_assimilate,
)

@dataclass
class PipelineState:
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

        state = PipelineState(
            models={
                "truth": model_truth,
                "ensemble_kf": copy.deepcopy(model_ensemble),
                "ensemble_qp": copy.deepcopy(model_ensemble),
            },
            histories={"kf": [], "qp": []},
        )

        for i in tqdm.tqdm(range(num_steps), desc="Pipeline Progress"):
            state.iteration = i
            state = self._forecast_truth(state)
            state = self._generate_observations(state)
            state = self._assimilate_and_forecast(state)

        if generate_evolution_animations:
            self._generate_animations(state)

        return state

    def _forecast_truth(self, state: PipelineState):
        """Step 1: Truth model step"""
        if state.iteration > 0:
            state.models["truth"].apply_nsub_steps()
        return state

    def _generate_observations(self, state: PipelineState):
        """Step 2: Observation generation from truth"""
        truth_state = state.models["truth"].get_current_state()

        state.observations = generate_observation(
            truth_state=truth_state,
            num_ensemble_members=self.num_ensemble_members,
            random_generator=self.rngs.obs_rng,
        )
        state.observation_locations = generate_radar_masks(
            state_truth=truth_state, random_generator=self.rngs.radar_rng
        )
        return state

    def _assimilate_and_forecast(self, state: PipelineState):
        """Step 3: Assimilate and forecast ensemble models"""
        ensemble_state = state.models["ensemble_kf"].get_current_state()

        kf_assimilated = kf_assimilate(
            ensemble=ensemble_state,
            observation=state.observations,
            observation_position=state.observation_locations,
        )
        qp_assimilated = qpens_assimilate(
            ensemble=ensemble_state,
            observation=state.observations,
            observation_position=state.observation_locations,
        )

        state.histories["kf"].append(kf_assimilated)
        state.histories["qp"].append(qp_assimilated)

        state.models["ensemble_kf"].apply_nsub_steps(kf_assimilated)
        state.models["ensemble_qp"].apply_nsub_steps(qp_assimilated)

        return state

    def _generate_animations(self, state: PipelineState):
        """Generate output animations"""
        state.models["ensemble_kf"].animate_evolution_from_history(
            state.histories["kf"], "./out/model_evolution_kf.mp4"
        )
        state.models["ensemble_qp"].animate_evolution_from_history(
            state.histories["qp"], "./out/model_evolution_qp.mp4"
        )
        state.models["truth"].animate_evolution(False)
