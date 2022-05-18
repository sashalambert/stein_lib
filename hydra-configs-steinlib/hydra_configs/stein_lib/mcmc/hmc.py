from dataclasses import dataclass, field
from typing import Any

@dataclass
class HMCConf:
    _target_: str = "stein_lib.mcmc.hmc.HMC"
    step_size: float = 2.5,
    num_steps_per_sample: int = 25
    num_restarts: int = 1

@dataclass
class NUTSConf:
    _target_: str = "stein_lib.mcmc.hmc.NUTS"
    step_size: float = 2.5,
    num_steps_per_sample: int = 25
    num_restarts: int = 1
    burn_in_steps: int = 100
