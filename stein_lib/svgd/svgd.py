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

import torch
import numpy as np
from time import time

from .composite_kernels import (
    iid,
)
from .LBFGS import FullBatchLBFGS
from ..utils import get_jacobian, calc_pw_distances, calc_scaled_pw_distances
from stein_lib.models.double_banana_analytic import doubleBanana_analytic


class SVGD():
    """
        Uses analytic kernel gradients.
    """

    def __init__(
            self,
            kernel=None,
            kernel_structure=None,
            verbose=False,
            control_dim=None,
            repulsive_scaling=1.,
            **kernel_params, #not sure if necessary anymore
    ):

        self.verbose = verbose
        self.kernel_base_type = kernel.__name__
        self.kernel_structure = kernel_structure
        self.ctrl_dim = control_dim
        self.repulsive_scaling = repulsive_scaling

        self.base_kernel = kernel(**kernel_params)
        self.kernel = self.get_kernel(**kernel_params)
        self.geom_metric_type = kernel_params['geom_metric_type']
        self.hessian_scaled = False
        self._M = None

        supported_base_kernels = [
            'ABCMeta', # not entirely sure why this is necessary yet
            'RBF',
            'IMQ',
            'RBF_Anisotropic',
            'Linear',
        ]
        hessian_scaled_kernels = [
            'RBF_Anisotropic',
            'RBF_Matrix',
            'IMQ_Matrix',
            'RBF_Weighted_Matrix',
        ]
        if self.kernel_base_type in hessian_scaled_kernels :
            self.hessian_scaled = True

        #TODO: check if this is semantically correct
        if self.kernel_base_type not in supported_base_kernels: 
            raise IOError('Stein kernel type not recognized: ', self.kernel_base_type)

    # this iid wrapper can be included when building kernel rather than apart of this class
    def get_kernel(
            self,
            **kernel_params,
    ):

        if self.kernel_structure is None:
            return self.base_kernel
        elif self.kernel_structure == 'iid':
            return iid(
                kernel=self.base_kernel,
                **kernel_params,
            )
        else:
            raise IOError('Kernel structure not recognized for SVGD: ',
                          self.kernel_structure,)

    def get_svgd_terms(
            self,
            X,
            dlog_p,
            M=None,
    ):
        """
        Parameters
        ----------
        X : Tensor
          Stein particles. Tensor of shape [batch, dim],
        dlog_p : Tensor
          Gradient of log probability. Shape [batch, dim]
        M : (Optional)
          Negative Hessian or Fisher matrices. Tensor of shape [batch, dim, dim]

        Returns
        -------
        grad: Tensor
            Attractive SVGD gradient term.
            Shape [batch, dim].
        rep: Tensor
            Repulsive SVGD term.
            Shape [batch, dim].
        pw_dists_sq: Tensor
            Squared pairwise distances between particles. Can be scaled by a
            metric.
            Shape [batch, batch].
        """

        k_XX, grad_k, pw_dists_sq = self.evaluate_kernel(X, M)

        grad = k_XX.mm(dlog_p) / k_XX.size(1)
        rep = grad_k.mean(1)

        return grad, rep, pw_dists_sq

    def evaluate_kernel(self, X, M=None):
        """

        Parameters
        ----------
        X :  tensor. Stein particles, of shape [batch, dim],
        M : (Optional) Negative Hessian or Fisher matrices. Tensor of shape [batch, dim, dim]

        Returns
        -------
        k_XX :
            tensor of shape [batch, batch]
        grad_k :
            tensor of shape [batch, batch, dim]
        pw_dists_sq:

        """
        k_XX, grad_k, _, pw_dists_sq = self.kernel.eval(
            X, X.clone().detach(),
            M,
            compute_dK_dK_t=False,
        )
        return k_XX, grad_k, pw_dists_sq
    
    def phi(
        self,
        X,
        dlog_p,
        dlog_lh=None,
        Hess=None,
        Hess_prior=None,
        Jacobian=None,
        copy_pw_dists=False
    ):
        """
        Computes the SVGD gradient.

        Parameters
        ----------
        X : Tensor
            Stein particles, of shape [batch, dim].
        dlog_p : Tensor
            Score function, of shape [batch, dim].

        Returns
        -------
        Phi: Tensor
            Empirical Stein gradient, of shape [batch, dim].
        pw_dists_sq: Tensor
            Squared pairwise distances between particles. Can be metric-scaled.
            Shape [batch, batch].
        """

        if self.geom_metric_type is None:
            M = None
            pass
        elif self.geom_metric_type == 'full_hessian':
            assert Hess is not None
            M = - Hess
        elif self.geom_metric_type == 'fisher':
            # Average Fisher matrix (likelihood only)
            np = dlog_lh.shape[0]
            M = torch.bmm(dlog_lh.reshape(np, -1, 1,), dlog_lh.reshape(np, 1, -1))
            # M -= torch.eye(M.shape[1], M.shape[2]) * 1.e-8
        elif self.geom_metric_type == 'jacobian_product':
            # Average Fisher matrix (full posterior gradient)
            M = torch.bmm(Jacobian.transpose(1, 2), Jacobian)
            M = M - Hess_prior
        elif self.geom_metric_type == 'riemannian':
            # Average Fisher matrix plus neg. Hessian of log prior
            b = dlog_lh.shape[0]
            Hess = torch.bmm(dlog_lh.view(b, -1, 1,), dlog_lh.view(b, 1, -1))
            M = - Hess - Hess_prior
        elif self.geom_metric_type == 'local_Hessians':
            # Average Fisher matrix plus neg. Hessian of log prior
            M = - Hess - Hess_prior
        else:
            raise NotImplementedError

        # SVGD attractive / repulsive terms, inter-particle distances
        grad, rep, pw_dists_sq = self.get_svgd_terms(
            X,
            dlog_p,
            M,
        )
        if self.verbose:
            print('gradient l2-norm: {:5.4f}'.format(
                grad.norm().detach().cpu().numpy()))
            print('repulsive l2-norm: {:5.4f}'.format(
                rep.norm().detach().cpu().numpy()))

        # SVGD gradient
        phi = grad + self.repulsive_scaling * rep

        self._pw_dists_sq = pw_dists_sq
        self._X = X

        return phi, pw_dists_sq

    def apply(
            self,
            X,
            model,
            iters=100,
            use_analytic_grads=False,
            optimizer=None,
    ):
        """
        Runs SVGD optimization on a distribution model, given a particle
         initialization X, and a selected optimization algorithm.

        Parameters
        ----------
        X : (Tensor)
            Stein particles, of shape [dim, num_particles]
        model:
            Probability distribution model instance. Can be of differentiable
            type torch.distributions, or custom model with analytic functions
            (see examples under 'stein_lib/models').
        iters:
            Number of optimization iterations.
        eps : Float
            Step size.
        use_analytic_grads: Bool
            Set to 'True' if probability model uses analytic gradients. If set to
            'False', numerical gradient will be computed.
        optimizer: torch.optim.Optimizer
            This should be initialized outside of svgd and passed to `apply`.
            Optimizer used for updates.
        """

        particle_history = []
        particle_history.append(X.detach().clone().cpu().numpy())

        dts = []
        X = torch.autograd.Variable(X, requires_grad=True)

        # reset optimizer params to empty and add particles as target parameters
        optimizer.param_groups[0]['params'] = []
        optimizer.param_groups[0]['params'].append(X)

        # Optimizer type
        def closure():
            optimizer.zero_grad()
            Hess = None
            if use_analytic_grads:

                if isinstance(model, doubleBanana_analytic):
                    # Used only by double_banana model
                    F = model.forward_model(X)
                    J = model.jacob_forward(X)
                    dlog_p = model.grad_log_p(X, F, J)
                else:
                    dlog_p = model.grad_log_p(X)

                if self.hessian_scaled and \
                    self.geom_metric_type not in ['fisher']:

                    if isinstance(model, doubleBanana_analytic):
                        ## Used only by double_banana model
                        # Gauss-Newton approximation
                        Hess = model.hessian(X, J)  # returns hessian of negative log posterior
                    else:
                        Hess = model.hessian(dlog_p, X)
            else:
                # Numerical Gradients
                log_p = model.log_prob(X).unsqueeze(1)
                dlog_p = torch.autograd.grad(
                    log_p.sum(),
                    X,
                    create_graph=True,
                )[0]
                if self.hessian_scaled and \
                    self.geom_metric_type not in ['fisher']:
                    Hess = get_jacobian(dlog_p, X)

            # SVGD gradient
            with torch.no_grad():
                Phi, pw_dists_sq = self.phi(
                    X,
                    dlog_p,
                    dlog_lh=dlog_p,
                    Hess=Hess,
                )
            X.grad = -1. * Phi
            # check(X.grad, 'X.grad')
            loss = 1.
            return loss

        for i in range(iters):
            self.i = i
            t_start = time()
            if isinstance(optimizer, FullBatchLBFGS):
                options = {'closure': closure, 'current_loss': closure()}
                optimizer.step(options)
            else:
                optimizer.step(closure)
            dt = time() - t_start
            if self.verbose:
                print('dt (SVGD): {}\n'.format(dt))
            dts.append(dt)
            particle_history.append(X.clone().detach().cpu().numpy())
        dt_stats = np.array(dts)
        if self.verbose:
            print("\nAvg. SVGD compute time: {}".format(dt_stats.mean()))
            print("Std. dev. SVGD compute time: {}\n".format(dt_stats.std()))

        (pw_dists,
         pw_dists_scaled,) = self.get_pairwise_dists()

        return (
            X,
            particle_history,
            pw_dists,
            pw_dists_scaled,
        )

    def get_pairwise_dists(self):
        # pw_dists output from svgd-gradient computation
        pw_dists_out = torch.sqrt(self._pw_dists_sq.clone().detach())
        X = self._X.clone().detach()

        if self.hessian_scaled:
            # Hessian-scaled pw_dists
            pw_dists_scaled = pw_dists_out
            pw_dists = calc_pw_distances(X)
        else:
            # Euclidean Pairwise distances
            pw_dists = pw_dists_out
            pw_dists_scaled = None
        return pw_dists, pw_dists_scaled

def check(tsr, name):
    """Check a tensor for inf/nan/large values."""
    isinf = torch.isinf(tsr)
    if isinf.any():
        if isinf.all():
            infind = 'all'
        else:
            infind = torch.nonzero(isinf)
        print(name, 'isinf', infind, flush=True)

    isnan = torch.isnan(tsr)
    if isnan.any():
        if isnan.all():
            nanind = 'all'
        else:
            nanind = torch.nonzero(isnan)
        print(name, 'isnan', nanind, flush=True)

    if (tsr.abs() > 1e6).any():
        print(name, 'isvlarge', flush=True)
