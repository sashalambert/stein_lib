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
from stein_lib.svgd.svgd import SVGD
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

# import sys
# np.set_printoptions(threshold=sys.maxsize)


class RBF_kernel:
    """
        k(x, x') = exp( - (x - y) M (x - y)^T / (2 * d))
    """
    def __init__(
        self,
        hessian_scale=1,
        median_heuristic=True,
        **kwargs,
    ):

        self.hessian_scale = hessian_scale
        self.median_heuristic = median_heuristic

    def eval(self, X, Y, M, **kwargs):
        """
        Parameters
        ----------
        X : Tensor of shape [batch, dim],
        Y : Tensor of shape [batch, dim],
        M : Tensor of shape [dim, dim],

        Returns
        -------
        """
        assert X.shape == Y.shape

        # PSD stabilization
        # M_psd = 0.5 * (M + M.T)

        M *= self.hessian_scale
        b, d = X.shape
        diff_XY = X.unsqueeze(1) - Y   # b x b x d
        diff_XY = diff_XY.reshape(b, b, 1, d)
        diff_XY_M = diff_XY @ M # (b, b, d)

        pw_dists_sq = diff_XY_M @ diff_XY.reshape(b, b, d, 1)
        pw_dists_sq = pw_dists_sq.reshape(b, b)

        if self.median_heuristic:
            h = torch.median(pw_dists_sq).detach()
            h = h / np.log(X.shape[0])
        else:
            h = self.hessian_scale * d

        K = (- 0.5 * pw_dists_sq / h).exp()
        # K = (- pw_dists_sq / h).exp()

        d_K_Xi = K.unsqueeze(2) * diff_XY_M.reshape(b, b, d) / h
        return (
            K,
            d_K_Xi,
        )


class MatrixMixtureSVGD(SVGD):

    def __init__(
            self,
            kernel_base_type='RBF_Matrix',
            **kwargs,
    ):
        super().__init__(kernel_base_type, **kwargs)

    def get_base_kernel(
            self,
            **kernel_params,
    ):
        # if self.kernel_base_type == 'RBF_Matrix':
        #     return RBF_Matrix(
        #         **kernel_params,
        #     )
        if self.kernel_base_type == 'RBF_Matrix':  # Does not precondition
            return RBF_kernel(
                **kernel_params,
            )
        else:
            raise IOError('Weighted-Matrix-SVGD kernel type not recognized: ',
                          self.kernel_base_type)

    def get_kernel(
            self,
            **kernel_params,
    ):

        if self.kernel_structure is None:
            return self.base_kernel
        else:
            raise IOError('Kernel structure not recognized for matrix-SVGD: ',
                          self.kernel_structure,)

    def get_pairwise_dists_sq(self, X, Y, M):
        """
        Get batched pairwise-distances squared.
        Parameters
        ----------
        X : Tensor of shape [batch, dim],
        Y : Tensor of shape [batch, dim],
        M : Tensor of shape [batch, dim, dim],

        Returns
        -------
        pw_dists : Tensor of shape [batch, batch]
        ,
        """
        b, d = X.shape
        diff_XY = X.unsqueeze(1) - Y   # b x b x d
        diff_XY = diff_XY.reshape(b, b, 1, d)
        diff_XY_M = diff_XY @ M # (b, b, d)
        pw_dists_sq = diff_XY_M @ diff_XY.reshape(b, b, d, 1)
        return pw_dists_sq.squeeze(), diff_XY_M.reshape(b, b, d)

    def get_weights(
            self,
            X,
            H,
            H_diff,
            pw_dists_sq,
    ):
        ## Debug: test using average metric for weights
        # M = H.mean(0)
        # b, d = X.shape
        # Y = X.clone()
        # diff_XY = X.unsqueeze(1) - Y   # b x b x d
        # diff_XY = diff_XY.reshape(b, b, 1, d)
        # diff_XY_M = diff_XY @ M # (b, b, d)
        #
        # pw_dists_sq = diff_XY_M @ diff_XY.reshape(b, b, d, 1)
        # pw_dists_sq = pw_dists_sq.reshape(b, b)
        # H_diff = diff_XY_M.reshape(b, b, d)

        # TODO: neg. definite H
        ww = torch.exp( - 0.5 * ( pw_dists_sq - pw_dists_sq.min(0).values - torch.logdet(H)) )
        # ww = torch.exp( - 0.5 * (pw_dists_sq - pw_dists_sq.min(0).values) )
        w = ww / torch.sum(ww, dim=0)
        dlog_w = torch.sum((H_diff[:,None,:,:] - H_diff[None,:,:,:]) * ww[None,:,:,None], dim=1)
        dlog_w = dlog_w / torch.sum(ww, dim=0)[None,:,None]
        return w, dlog_w

    def weighted_Hessian_SVGD(self, X, dlog_p, Hess, H_inv, w):

        # k_XX, grad_k = self.kernel.eval(X, X, H_inv)
        k_XX, grad_k = self.kernel.eval(X, X, Hess)

        # print('k_XX', k_XX)
        # print('grad_k', grad_k)

        velocity = torch.sum(
            w[None,:,None] * k_XX[:,:,None] * dlog_p[None,:,:],
            dim=1
        ) + torch.sum(
            w[:,None,None] * grad_k,
            dim=0
        )
        velocity = velocity @ H_inv
        return velocity

    def get_update(
            self,
            X,
            dlog_p,
            Hess,
            Hess_prior=None,
            # alpha=0.5,
            alpha=0.,
    ):

        """
        Handle matrix-valued SVGD terms.

        Parameters
        ----------
        X :  Stein particles. Tensor of shape [batch, dim],
        dlog_p : tensor of shape [batch, dim]
        M : Negative Hessian or Fisher matrices. Tensor of shape [batch, dim, dim]

        Returns
        -------
        gradient: tensor of shape [batch, dim]
        repulsive: tensor of shape [batch, dim]

        """

        b, d = X.shape

        Hess_avg = Hess.mean(0)
        Hess = alpha * Hess + (1 - alpha) * Hess_avg # for 'robustness'

        pw_dists_sq, H_diff = self.get_pairwise_dists_sq(X, X, Hess)

        w, dlog_w = self.get_weights(X, Hess, H_diff, pw_dists_sq)

        # print('\nw', w)
        # print('dlog_w', dlog_w)
        H_inv = torch.inverse(Hess)  # b, d, d
        # print('H_inv', H_inv)

        velocity = torch.zeros_like(X)

        for i in range(b):
            velocity += w[i, :, None] * self.weighted_Hessian_SVGD(
                X,
                dlog_p + dlog_w[i,:,:],
                Hess[i,:,:],
                H_inv[i,:,:],
                w[i,:],
            )
            # ) / b
        # print("velocity", velocity)
        return velocity


    def phi(
        self,
        X,
        dlog_p,
        dlog_lh,
        Hess=None,
        Hess_prior=None,
        reshape_inputs=True,
        transpose=False,
    ):
        """
        Parameters
        ----------
        X : (Tensor)
            Stein particles, of shape [num_particles, dim],
            or of shape [dim, num_particles]. If Tensor dimension is greater than 2,
            extra dimensions will be flattened.
        dlog_p : (Tensor)
            Score function, of shape [num_particles, dim]
            or of shape [dim, num_particles]. If Tensor dimension is greater than 2,
            extra dimensions will be flattened.
        transpose: Bool
            Transpose input and output Tensors.
        Returns
        -------
        Phi: (Tensor)
            Empirical Stein gradient, of shape [num_particles, dim]
        """

        shape_original = X.shape
        if reshape_inputs:
            X, dlog_p, Hess = self.reshape_inputs(
                X,
                dlog_p,
                Hess,
                transpose,
            )

        if self.geom_metric_type is None:
            pass
        if self.geom_metric_type == 'full_hessian':
            assert Hess is not None
            M = - Hess
            # M += 1.e-6 * torch.eye(M.shape[1], M.shape[2])
        elif self.geom_metric_type == 'fisher':
            ## Average Fisher matrix (likelihood only)
            np = dlog_lh.shape[0]
            M = torch.bmm(dlog_lh.reshape(np, -1, 1,), dlog_lh.reshape(np, 1, -1))
            M += 1.e-6 * torch.eye(M.shape[1], M.shape[2])
        elif self.geom_metric_type == 'jacobian_product':
            ## Average Fisher matrix (full posterior gradient)
            dim = dlog_p.shape[-1]
            M = torch.bmm(dlog_p.view(-1, dim, 1,), dlog_p.view(-1, 1, dim))
            M += 1.e-6 * torch.eye(M.shape[1], M.shape[2])
        elif self.geom_metric_type == 'riemannian':
            # Average Fisher matrix plus neg. Hessian of log prior
            b = dlog_lh.shape[0]
            F = torch.bmm(dlog_lh.view(b, -1, 1,), dlog_lh.view(b, 1, -1))
            Hess_prior = Hess_prior.reshape(
                        dlog_p.shape[0],
                        dlog_p.shape[1],
                        dlog_p.shape[1],
                    )
            M = F - Hess_prior
        else:
            raise NotImplementedError

        phi = self.get_update(
            X,
            dlog_p,
            M,
        )
        # Reshape Phi to match original input tensor dimensions
        if reshape_inputs:
            if transpose:
                phi = phi.t()
            phi = phi.reshape(shape_original)

        return phi