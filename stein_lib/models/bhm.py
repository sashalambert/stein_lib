import numpy as np
import torch
from pathlib import Path
from bhmlib.BHM.pytorch.bhmtorch_cpu import BHM2D_PYTORCH


class BayesianHilbertMap:
    def __init__(
            self,
            file_path=None,
            limits=((-10, 20,), (-25, 5)),
            # limits=torch.tensor([[-8, 18,], [-23, 3]]),
    ):

        # Load trained Bayesian Hilbert Map
        params = torch.load(file_path)
        self.bhm = BHM2D_PYTORCH(torch_kernel_func=True, **params)
        self.limits = torch.tensor(limits)

    def log_prob(self, x):
        log_p = self.bhm.log_prob_vacancy(x)
        if self.limits is not None:
            scale = 1.
            log_p -= torch.exp(-scale*(x[:, 0]  - self.limits[0, 0]))
            log_p -= torch.exp( scale*(x[:, 0]  - self.limits[0, 1]))
            log_p -= torch.exp(-scale*(x[:, 1]  - self.limits[1, 0]))
            log_p -= torch.exp( scale*(x[:, 1]  - self.limits[1, 1]))
        return log_p

    def grad_log_p(self, x):
        return self.bhm.grad_log_p_vacancy(x)


if __name__ == '__main__':

    import bhmlib
    bhm_path = Path(bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter010.pt'

    bhm = BayesianHilbertMap(model_file)

