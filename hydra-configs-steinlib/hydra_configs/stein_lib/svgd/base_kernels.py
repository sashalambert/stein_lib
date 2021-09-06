from dataclasses import dataclass, field
from typing import Any

@dataclass
class BaseKernelConf:
    _target_: str = "stein_lib.svgd.base_kernels.BaseKernel"
    analytic_grad: bool = False

@dataclass
class IMQConf(BaseKernelConf):
    _target_: str = "stein_lib.svgd.base_kernels.IMQ"
    alpha: float = 1.0
    beta: float = -0.5
    hessian_scale: float = 1.0
    analytic_grad: bool = True
    median_heuristic: bool = False

@dataclass
class LinearConf(BaseKernelConf):
    _target_: str = "stein_lib.svgd.base_kernels.Linear"
    analytic_grad: bool = True
    subtract_mean: bool = True
    with_scaling: bool = False

@dataclass
class RBFConf(BaseKernelConf):
    _target_: str = "stein_lib.svgd.base_kernels.RBF"
    bandwidth: float = -1.0
    analytic_grad: bool = True

@dataclass
class RBF_AnisotropicConf(BaseKernelConf):
    _target_: str = "stein_lib.svgd.base_kernels.RBF_Anisotropic"
    hessian_scale: float = 1.0
    analytic_grad: bool = True
    median_heuristic: bool = False

