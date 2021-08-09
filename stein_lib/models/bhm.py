import numpy as np
import torch

class BayesianHilbertMap:

    def __init__(
            self,
    ):

        self.dist = MixtureOfDiagNormals(mus, sigmas, mix_coeffs)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def grad_log_p(self, x):
        x_ = torch.autograd.Variable(x, requires_grad=True)
        dlog_p = torch.autograd.grad(
            self.log_prob(x_).sum(),
            x_,
        )[0]
        return dlog_p