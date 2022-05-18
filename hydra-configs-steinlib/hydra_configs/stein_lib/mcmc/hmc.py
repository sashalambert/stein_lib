from dataclasses import dataclass, field
from typing import Any

@dataclass
class HMCConf:
    _target_: str = "stein_lib.mcmc.hmc.HMC"
    sampler_type: str = 'nuts'
    step_size: float = 2.5,
    num_steps_per_sample: int = 25
    burn_in_steps: int = 100
    num_restarts: int = 1