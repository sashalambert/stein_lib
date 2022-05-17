"""
Copyright (c) 2020-2021 Alexander Lambert

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
from pathlib import Path
from bhmlib.BHM.pytorch.bhmtorch_cpu import BHM2D_PYTORCH


class BayesianHilbertMap:
    def __init__(
            self,
            file_path=None,
            limits=((-10, 20,), (-25, 5)),
            device=None,
    ):

        self.device = device
        # Load trained Bayesian Hilbert Map
        params = torch.load(file_path)
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                params[k] = v.to(device)
        self.bhm = BHM2D_PYTORCH(torch_kernel_func=True, **params)
        self.limits = torch.tensor(limits).to(device)

    def log_prob(self, x):
        log_p = self.bhm.log_prob_vacancy(x)
        if self.limits is not None:
            scale = 1.
            log_p -= torch.exp(-scale*(x[:, 0] - self.limits[0, 0]))
            log_p -= torch.exp( scale*(x[:, 0] - self.limits[0, 1]))
            log_p -= torch.exp(-scale*(x[:, 1] - self.limits[1, 0]))
            log_p -= torch.exp( scale*(x[:, 1] - self.limits[1, 1]))
        return log_p

    def grad_log_p(self, x):
        return self.bhm.grad_log_p_vacancy(x)

if __name__ == '__main__':

    import bhmlib
    bhm_path = Path(bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter010.pt'

    bhm = BayesianHilbertMap(model_file)

