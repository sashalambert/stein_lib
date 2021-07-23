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
from abc import ABC, abstractmethod


class BaseMatrixKernel(ABC):

    def __init__(
        self,
        analytic_grad=True,
        median_heuristic=False,
    ):
        self.analytic_grad = analytic_grad
        self.median_heuristic = median_heuristic

    @abstractmethod
    def eval(self, X, Y, M=None, **kwargs):
        """
        Evaluate kernel function and corresponding gradient terms for batch of inputs.

        Parameters
        ----------
        X : Tensor
            Data, of shape [batch, dim]
        Y : Tensor
            Data, of shape [batch, dim]
        M : Tensor (Optional)
            Metric, of shape [batch, dim, dim]
        kwargs : dict
            Kernel-specific parameters

        Returns
        -------
        K: Tensor
            Kernel Gram matrix which is pre-conditioned by the inverse metric M^-1.
             Of shape [batch, batch, dim, dim].
        d_K_Xi: Tensor
            Kernel gradients wrt. first input X. Shape: [batch, batch, dim]
        """
        pass


class RBF_Matrix(BaseMatrixKernel):
    """
        RBF Matrix-valued kernel, with averaged Hessian/Fisher metric M.
        Similar to the RBF_Anisotropic kernel, but preconditioned with inverse of M.

        k(x, x') = M^-1 exp( - (x - y) M (x - y)^T / (2 * d))
    """
    def __init__(
        self,
        hessian_scale=1,
        analytic_grad=True,
        median_heuristic=False,
        **kwargs,
    ):
        super().__init__(
            analytic_grad,
            median_heuristic,
        )
        self.hessian_scale = hessian_scale

    def eval(self, X, Y, M=None, **kwargs):

        assert X.shape == Y.shape
        b, dim = X.shape

        # Empirical average of Hessian / Fisher matrices
        M = M.mean(dim=0)

        # PSD stabilization
        M_psd = 0.5 * (M + M.T)

        M *= self.hessian_scale
        X_M_Xt = X @ M @ X.t()
        X_M_Yt = X @ M @ Y.t()
        Y_M_Yt = Y @ M @ Y.t()

        pairwise_dists_sq = -2 * X_M_Yt + X_M_Xt.diag().unsqueeze(1) + Y_M_Yt.diag().unsqueeze(0)
        if self.median_heuristic:
            h = torch.median(pairwise_dists_sq).detach()
            h = h / np.log(X.shape[0])
            # h *= 0.5
        else:
            # h = self.hessian_scale * X.shape[1]
            h = self.hessian_scale
        # bandwidth = self.hessian_scale * X.shape[1]
        # K = (- pairwise_dists_sq / h).exp()
        K = (- 0.5 * pairwise_dists_sq / h).exp()
        d_K_Xi = K.unsqueeze(2) * ( (X.unsqueeze(1) - Y) @ M ) * 2 / h

        ## Matrix preconditioning
        M_inv = torch.inverse(M)

        K = K.reshape(b, b, 1, 1)

        K = M_inv * K
        d_K_Xi = (M_inv @ d_K_Xi.unsqueeze(-1)).squeeze(-1)
        return (
            K,
            d_K_Xi,
        )

class IMQ_Matrix(BaseMatrixKernel):
    """
        IMQ Matrix-valued kernel, with metric M.
        k(x, x') = M^-1 (alpha + (x - y) M (x - y)^T ) ** beta
    """
    def __init__(
        self,
        alpha=1,
        beta=-0.5,
        hessian_scale=1,
        analytic_grad=True,
        median_heuristic=False,
        **kwargs,
    ):

        self.alpha = alpha
        self.beta = beta

        super().__init__(
            analytic_grad,
            median_heuristic,
        )
        self.hessian_scale = hessian_scale

    def eval(self, X, Y, M=None, **kwargs):

        assert X.shape == Y.shape
        b, dim = X.shape

        # Empirical average of Hessian / Fisher matrices
        M = M.mean(dim=0)

        # PSD stabilization
        M_psd = 0.5 * (M + M.T)

        M *= self.hessian_scale
        X_M_Xt = X @ M @ X.t()
        X_M_Yt = X @ M @ Y.t()
        Y_M_Yt = Y @ M @ Y.t()

        pairwise_dists_sq = -2 * X_M_Yt + X_M_Xt.diag().unsqueeze(1) + Y_M_Yt.diag().unsqueeze(0)
        if self.median_heuristic:
            h = torch.median(pairwise_dists_sq).detach()
            h = h / np.log(X.shape[0])
            # h *= 0.5
        else:
            h = self.hessian_scale * X.shape[1]
        # bandwidth = self.hessian_scale * X.shape[1]

        # K = (- pairwise_dists_sq / h).exp()
        # d_K_Xi = K.unsqueeze(2) * ( (X.unsqueeze(1) - Y) @ M) * 2 / h

        K = (( self.alpha + pairwise_dists_sq) ** self.beta).reshape(-1, 1)
        d_K_Xi = self.beta * ((self.alpha + pairwise_dists_sq) ** (self.beta - 1)).unsqueeze(2) \
                 * ( -1. * (X.unsqueeze(1) - Y) @ M ) * 2 / h

        ## Matrix preconditioning
        M_inv = torch.inverse(M)

        K = K.reshape(b, b, 1, 1)

        K = M_inv * K
        d_K_Xi = (M_inv @ d_K_Xi.unsqueeze(-1)).squeeze(-1)
        return (
            K,
            d_K_Xi,
        )


class RBF_Weighted_Matrix(BaseMatrixKernel):

    def __init__(
        self,
        hessian_scale=1,
        analytic_grad=True,
        alpha=0.5,
        **kwargs,
    ):
        super().__init__(
            analytic_grad,
        )
        self.hessian_scale = hessian_scale
        self.alpha = alpha

    def get_mix_weights(self, X, M, pw_dist_sq):
        """
        Finds the Gaussian mixture weights used by the weighted Kernel.

        Parameters
        ----------
        X : Tensor
            Input values, of shape [batch,  dim]
        M : Tensor
            Metric tensor, of shape [batch, dim, dim]
        pw_dist_sq : Tensor
            Pair-wise distances for each metric tensor, of shape [batch, batch, batch]
            Last dimension corresponds to each metric.

        Returns
        -------
        mix_weights: Tensor
            Gaussian mixture weights, of shape [batch, batch]
        mix_dlog_w: Tensor
            Log-derivative of weights, of shape [batch, batch, dim]
        """

        # Get pw_dists for z_el and corresponding M_el
        pw_dist_sq_el = torch.diagonal(pw_dist_sq, dim1=1, dim2=2)
        mix_weights = torch.softmax( - pw_dist_sq_el - torch.logdet(M), dim=0)

        #TODO: parallelize this somehow?
        mix_dlog_w = [
            torch.autograd.grad(
                mix_weights[i].sum(),
                X,
                retain_graph=True,
            )[0] for i in range(mix_weights.shape[0])
        ]
        mix_dlog_w = torch.stack(mix_dlog_w, dim=0)

        return mix_weights, mix_dlog_w

    def eval(self, X, Y, M=None, **kwargs):

        assert X.shape == Y.shape

        M = 0.5 * (M + M.transpose(1, 2)) # PSD stabilization

        M *= self.hessian_scale # M of shape (batch, dim, dim)

        # Mix w/ average Hessian for robustness (from Wang et al. 2019 implementation)
        M = (1 - self.alpha) * M.mean(0) + self.alpha * M

        bandwidth = self.hessian_scale * X.shape[1]

        b, dim = X.shape
        diff_XY = X.unsqueeze(1) - Y # (b, b, dim)
        diff_XY = diff_XY.reshape(b, b, 1, 1, dim)
        diff_XY_M = diff_XY @ M # (b, b, b, dim)
        pairwise_dists_sq = diff_XY_M @ diff_XY.transpose(-2, -1)

        # Mixture weights
        w, dlog_w = self.get_mix_weights(X, M, pairwise_dists_sq.reshape(b, b, b))
        w = w.unsqueeze(-1)
        w_w_T = w.bmm(w.transpose(1, 2))

        # Mixture Grammians and gradients
        M_inv = torch.inverse(M)
        K_el = (- pairwise_dists_sq / bandwidth).exp()
        K_el = M_inv * K_el # (b, b, b, d, d)
        d_K_Xi_el = (diff_XY_M @ K_el) * 2 / bandwidth

        # Full Kernel Grammian
        w_w_T = w_w_T.reshape(b, b, b, 1, 1)
        K = (w_w_T * K_el).sum(dim=2)

        # Full Kernel gradient
        dlog_w_K_el = torch.einsum('abcde, bce -> abcd', K_el, dlog_w)
        w_w_T = w_w_T.reshape(b, b, b, 1)
        dlog_w_K_el = dlog_w_K_el.reshape(b, b, b, dim)
        d_K_Xi_el = d_K_Xi_el.reshape(b, b, b, dim)
        d_K_Xi = (w_w_T * (dlog_w_K_el + d_K_Xi_el)).sum(dim=2)

        return (
            K,
            d_K_Xi,
        )
