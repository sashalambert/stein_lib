import numpy as np
import torch
from pathlib import Path
from bhmlib.BHM.pytorch.bhmtorch_cpu import BHM2D_PYTORCH

class BayesianHilbertMap:
    def __init__(
            self,
            file_path=None,
    ):

        # Load trained Bayesian Hilbert Map
        params = torch.load(file_path)
        self.bhm = BHM2D_PYTORCH(**params)

    def log_prob(self, x):
        return self.bhm.log_prob(x)

    def grad_log_p(self, x):
        x_ = torch.autograd.Variable(x, requires_grad=True)
        dlog_p = torch.autograd.grad(
            self.log_prob(x_).sum(),
            x_,
        )[0]
        return dlog_p


if __name__ == '__main__':

    import bhmlib
    bhm_path = Path(bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter010.pt'

    bhm = BayesianHilbertMap(model_file)
