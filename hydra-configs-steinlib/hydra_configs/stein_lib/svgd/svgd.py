from dataclasses import dataclass, field
from typing import Any

from hydra_configs.stein_lib.svgd.base_kernels import BaseKernelConf, RBF_AnisotropicConf

@dataclass
class SVGDConf:
    _target_: str = "stein_lib.svgd.svgd.SVGD"
    verbose: bool = False
    control_dim: Any = None
    repulsive_scaling: float = 1.0
    kernel: BaseKernelConf = RBF_AnisotropicConf()
    kernel_structure: Any = None
    geom_metric_type: str = "fisher"
