from dataclasses import dataclass, field
from typing import Any

@dataclass
class LDConf:
    _target_: str = "stein_lib.mcmc.sgld.LangevinDynamics"
    lr: float = 0.1
    lr_final: float = 1.e-2
    max_itr: int = 1
    gamma: float = -0.55

@dataclass
class MALAConf:
    _target_: str = "stein_lib.mcmc.langevin.MetropolisAdjustedLangevin"
    lr: float = 0.1
    lr_final: float = 1.e-2
    max_itr: int = 1
    gamma: float = -0.55
