from dataclasses import dataclass, field
from typing import Any

@dataclass
class SGLDConf:
    _target_: str = "stein_lib.mcmc.sgld.LangevinDynamics"
    data: Any = None
    model: Any = None
    lr: float = 0.1
    lr_final: float = 1.e-2
    max_itr: int = 1
