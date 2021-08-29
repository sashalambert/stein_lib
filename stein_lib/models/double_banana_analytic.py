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

class doubleBanana_analytic:
    """
        Bi-modal posterior distribution with analytic derivative terms.
    """
    def __init__(
            self,
            mu_n=0.,
            seed=0,
            prior_var=1.,
            obs_var=0.3**2,
            # a=1,
            a=0,
            b=100,
    ):

        self.dim = 2
        self.a = a
        self.b = b
        # Prior prob. params
        self.mu_0 = torch.zeros((self.dim, 1))
        self.var_0 = prior_var * torch.ones((self.dim, 1))

        # Likelihood prob. params
        self.mu_n = mu_n
        self.var_n = obs_var

        torch.manual_seed(seed)

        self.thetaTrue = np.random.normal(size=self.dim)

    def forward_model(self, x):
        """
        Observation function.

        Parameters
        ----------
        x : (Tensor)
            Tensor of 2D samples, with shape [2, num_samples]

        Returns
        -------
        F : (Tensor)
            Tensor of 1-D function values, with shape [1, num_samples]
        """
        assert x.dim() == 2 and x.shape[0] == self.dim
        return torch.log( ( self.a - x[0] )**2 + self.b * ( x[1] - x[0]**2 )**2 )

    def log_lh(self, x, F=None):
        """
        Returns the log-likelihood probability densities.

        Parameters
        ----------
        x : (Tensor)
            Tensor of 2D samples, with shape [2, num_samples]
        F : (Tensor)
            (Optional) Function evaluations of samples, F = F(x). With shape
             [1, num_samples]

        Returns
        -------
        log_prob:(Tensor)
            Tensor of observation probabilities, with shape [1, num_samples]
        """
        if F is None:
            F = self.forward_model(x)
        F = F.reshape(1, -1)
        return - 0.5 * torch.sum( (self.mu_n - F) ** 2, dim=0) / self.var_n

    def jacob_forward(self, x):
        """
        Jacobian of the forward model.

        Parameters
        ----------
        x : (Tensor)
            Tensor of 2D samples, with shape [2, num_samples]

        Returns
        -------
        J : (Tensor)
            Jacobian tensor, with shape [2, num_samples]
        """
        J = torch.stack( (
            (2 * ( x[0, :] - self.a - 2 * self.b * x[0, :] * (x[1, :] - x[0, :] ** 2) ) ) \
            / ( 1 + x[0, :] ** 2 - 2 * x[0, :] + self.b * (x[1, :] - x[0, :] ** 2) ** 2 ),
            ( 2 * self.b * (x[1, :] - x[0, :] ** 2) ) \
            / ( 1 + x[0, :] ** 2 - 2 * x[0, :] + self.b * (x[1, :] - x[0, :] ** 2) ** 2 )
        ) )
        return J

    def grad_log_lh(self, x, F=None, J=None):
        """
        Gradient of the log likelihood.
        """
        if F is None:
            F = self.forward_model(x).reshape(1,-1)
        if J is None:
            J = self.jacob_forward(x)
        return - J * (F - self.mu_n) / self.var_n

    def log_prior(self, x):
        """
        Returns the log-prior probability densities.

        Parameters
        ----------
        x : (Tensor)
            Tensor of 2D samples, with shape [2, num_samples]
        Returns
        -------

        """
        return - 0.5 * torch.sum( (x - self.mu_0) ** 2 / self.var_0, dim=0)

    def grad_log_prior(self, x):
        """
        Gradient of the log prior.
        """
        return - (x - self.mu_0) / self.var_0

    def log_prob(self, x):
        """
        Returns the log-posterior probability densities, without
        log_partition term.
        Parameters
        ----------
        x : (Tensor)
            Tensor of 2D samples, with shape [2, num_samples]
        Returns
        -------
        """
        return self.log_prior(x) + self.log_lh(x)

    def grad_log_p(self, x, F=None, J=None):
        """
        Gradient of the log posterior.
        """
        return self.grad_log_prior(x) + self.grad_log_lh(x, F, J)

    def hessian(self, x, J=None):
        """
        Gauss-Newton Hessian approximation of the log posterior.

        Parameters
        ----------
        x : (Tensor)
            Tensor of 2D samples, with shape [2, num_samples]

        J : (Tensor)
            Jacobian of the forward model, with shape [
        """
        if J is None:
            J = self.jacob_forward(x)

        Hess = J.reshape(self.dim, 1, -1) * J.reshape(1, self.dim, -1) / self.var_n \
              + (torch.eye(self.dim) / self.var_0).unsqueeze(2)
        return -1. * Hess


# num_particles = 100
#
# prior_dist = Normal(loc=0., scale=1.)
# particles_0 = prior_dist.sample((2, num_particles))
#
# particles = torch.autograd.Variable(
#             particles_0,
#             requires_grad=True,
#         )
#
