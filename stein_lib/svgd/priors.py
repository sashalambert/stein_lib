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

import torch.distributions as dist
import torch
from abc import ABC, abstractmethod
import numpy as np
from svmpc_np.utils import from_np


class mixture_of_gaussians:
    def __init__(
            self,
            means,
            sigmas,
            weights,
    ):
        """

        Parameters
        ----------
        means :
            shape: [num_particles, steps, ctrl_dim]
        sigmas :
            shape: [num_particles, steps, ctrl_dim]
        weights :
            shape: [num_particles]
        """
        self.num_particles, self.rollout_steps, self.ctrl_dim = means.shape
        components = dist.Independent(dist.Normal(means, sigmas), 2)
        mixture = dist.Categorical(weights)
        self.dist = dist.mixture_same_family.MixtureSameFamily(mixture, components)

    def sample(self, num_samples):
        return self.dist.sample(num_samples).transpose(0, 1)

    def log_prob(self, x):
        return self.dist.log_prob(x)

def avg_ctrl_to_goal(
        state,
        target,
        rollout_steps,
        dt,
        max_ctrl=100,
        control_type='velocity',
):
    """ Average control from state to target. Control must be
    first/second-derivative in state/target space. Ex. velocity control for
    cartesian states"""
    assert control_type in [
        'velocity',
        'acceleration',
    ]
    if control_type == 'velocity':
        return (
                (target - state)/ rollout_steps / dt
        ).clamp(max=max_ctrl)
    else:
        state_dim = state.dim()
        pos_dim = int(state_dim / 2)
        return (
                (target[:pos_dim] - state[:pos_dim]) / rollout_steps / dt**2
        ).clamp(max=max_ctrl)

def get_indep_gaussian_prior(
        sigma_init,
        rollout_steps,
        control_dim,
        mu_init=None,
):

    mu = torch.zeros(
        rollout_steps,
        control_dim,
    )
    if mu_init is not None:
        mu[:, :] = mu_init

    sigma = torch.ones(
        rollout_steps,
        control_dim,
    ) * sigma_init

    return dist.Normal(mu, sigma)

def get_multivar_gaussian_prior(
        sigma,
        rollout_steps,
        control_dim,
        Sigma_type='indep_ctrl',
        mu_init=None,
):
    """
    :param sigma: standard deviation on controls
    :param control_dim:
    :param rollout_steps:
    :param Sigma_type: Covariance prior type, 'indep_ctrl': diagonal Sigma,
    'const_ctrl': const. ctrl Sigma.
    :param mu_init:
    :return: distribution with MultivariateNormal for each control dimension
    """
    assert Sigma_type in [
        'indep_ctrl',
        'const_ctrl',
    ], 'Invalid type for control prior dist.'

    mu = torch.zeros(
        rollout_steps,
        control_dim,
    )

    if mu_init is not None:
        mu[:, :] = mu_init

    if Sigma_type == 'const_ctrl':
        # Const-ctrl covariance
        Sigma_gen = const_ctrl_Sigma

    elif Sigma_type == 'indep_ctrl':
        # Isotropic covariance
        Sigma_gen = diag_Sigma
    else:
        raise IOError('Sigma_type not recognized.')

    Sigma = Sigma_gen(
        sigma,
        rollout_steps,
        control_dim,
    )

    # check_Sigma_is_valid(Sigma)

    return Gaussian_Ctrl_Dist(
        rollout_steps,
        control_dim,
        mu,
        Sigma,
    )

def diag_Sigma(sigma, length=None, ctrl_dim=None):
    """
      Time-independent diagonal covariance matrix. Assumes independence
      across control dimension.
    """
    Sigma = torch.eye(
        length,
    ).unsqueeze(-1).repeat(1, 1, ctrl_dim)

    if isinstance(sigma, list):
        Sigma = Sigma * torch.from_numpy(np.array(sigma)).float()**2
    else:
        Sigma = Sigma * sigma**2
    return Sigma

def const_ctrl_Sigma(
        sigma,
        length=None,
        ctrl_dim=None,
):
    """
      Constant-control covariance prior. Assumes independence across control
      dimension.
    """

    if isinstance(sigma, list):
        sigma = torch.from_numpy(np.array(sigma)).float()

    L = torch.tril(
        torch.ones(
            length,
            length-1,
        ), diagonal=-1,
    )
    LL_t = torch.matmul(
        L, L.transpose(0, 1)
    )

    LL_t += torch.ones(
        length,
        length,
    )

    Sigma = LL_t.unsqueeze(-1).repeat(1, 1, ctrl_dim) * sigma**2
    return Sigma

def check_Sigma_is_valid(Sigma):
    """
    Check determinant of Sigma to pre-empt potential numerical instability
    in Multi-variate Gaussian.
    For example, dist.logprob(x) >> 1.
    :param Sigma: covariance matrix (Tensor) of shape [rollout_length,
    rollout_length, control_dim]
    """
    Sigma_np = Sigma.cpu().numpy()
    for i in range(Sigma.shape[-1]):
        det = np.linalg.det(Sigma_np[:,:,i])
        if det < 1.e-7:
            raise ZeroDivisionError(
                'Covariance-determinant too small, potential for underflow.  '
                'Consider increasing sigma.'
            )

class Prior_Ctrl_Traj_Dist (ABC):
    """
    Prior distribution on control trajectories, Assumes independence across
    ctrl_dim.
    """
    def __init__(
            self,
            rollout_steps,
            ctrl_dim,
    ):
        self.rollout_steps = rollout_steps
        self.ctrl_dim = ctrl_dim
        self.list_ctrl_dists = []

    @abstractmethod
    def make_dist(self):
        """ Construct list of sampling distribution, one for each control
        dimension"""
        pass

    def log_prob(
            self,
            samples,
            cond_inputs=None,
    ):
        """
        :param samples: control samples of shape ( num_particles, rollout_steps,
         ctrl_dim)
        :return: log_probs, of shape (num_particles.)
        """
        assert samples.dim() == 3
        assert samples.size(1) == self.rollout_steps
        assert samples.size(2) == self.ctrl_dim
        num_particles = samples.size(0)

        log_probs = torch.zeros(
            num_particles,
        )

        for i in range(self.ctrl_dim):
            samp = samples[:, :, i]  # [num_particles, rollout_steps]
            log_probs += self.list_ctrl_dists[i].log_prob(
                samp
            )

        return log_probs

    def update_means(self, means):
        for i in range(self.ctrl_dim):
            self.list_ctrl_dists[i].loc = means[..., i].detach().clone()

    def sample(
            self,
            num_samples,
            cond_inputs=None,
    ):
        """
        :param num_particles: number of control particles
        :param cond_inputs: conditional input (not implemented)
        :return: control tensor, of size (rollout_steps, num_particles,
        ctrl_dim)
        """
        U_s = torch.empty(
            num_samples,
            self.rollout_steps,
            self.ctrl_dim,
        )
        for i in range(self.ctrl_dim):
            U_s[:, :, i] = self.list_ctrl_dists[i].sample(
                (num_samples,)
            )
        return U_s

class Gaussian_Ctrl_Dist(Prior_Ctrl_Traj_Dist):
    """
    Multivariate Gaussian distribution for each control dimension.
    """
    def __init__(
            self,
            rollout_steps,
            ctrl_dim,
            mu=None,
            Sigma=None,
    ):

        assert mu.size(0) == rollout_steps
        assert mu.size(1) == ctrl_dim
        super().__init__(
            rollout_steps,
            ctrl_dim,
        )
        self.mu = mu
        self.Sigma = Sigma
        self.list_ctrl_dists = []

        self.make_dist()

    def make_dist(self):
        for i in range(self.ctrl_dim):
            self.list_ctrl_dists.append(
                dist.MultivariateNormal(
                    self.mu[:, i],
                    covariance_matrix=self.Sigma[:,:,i]
                ))

class Gamma_Ctrl_Dist(Prior_Ctrl_Traj_Dist):
    def __init__(self):
        raise NotImplementedError

    def make_dist(self):
        raise NotImplementedError
