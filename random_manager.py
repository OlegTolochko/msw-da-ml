import numpy as np
from dataclasses import dataclass


@dataclass
class RandomGenerators:
    """Contains multiple independent random number generators"""

    truth_rng: np.random.Generator
    ensemble_rng: np.random.Generator
    obs_rng: np.random.Generator
    radar_rng: np.random.Generator

    @classmethod
    def from_seed(self, base_seed: int):
        """Creates independent RNGs from a base seed"""
        ss = np.random.SeedSequence(base_seed)
        child_seeds = ss.spawn(4)

        return self(
            truth_rng=np.random.default_rng(child_seeds[0]),
            ensemble_rng=np.random.default_rng(child_seeds[1]),
            obs_rng=np.random.default_rng(child_seeds[2]),
            radar_rng=np.random.default_rng(child_seeds[3]),
        )
