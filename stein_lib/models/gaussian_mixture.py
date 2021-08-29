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
from pyro.distributions.diag_normal_mixture import MixtureOfDiagNormals


class mixture_of_gaussians:

    def __init__(
            self,
            num_comp,
            mu_list,
            sigma_list,
    ):

        mus = torch.from_numpy(np.array(mu_list))
        sigmas = torch.from_numpy(np.array(sigma_list))

        mix_prior = 1./ num_comp
        mix_coeffs = torch.ones(num_comp) * mix_prior

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

    def sample(self, s_shape):
        return self.dist.sample(s_shape)
