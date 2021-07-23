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
from abc import ABC, abstractmethod


class BaseKernel(ABC):

    def __init__(
        self,
        analytic_grad=True,
    ):

        self.analytic_grad = analytic_grad

    @abstractmethod
    def eval(self, X, Y, M=None, compute_dK_dK_t=False, **kwargs):
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
        compute_dK_dK_t : Bool
            Compute outer-products of kernel gradients.
        kwargs : dict
            Kernel-specific parameters

        Returns
        -------
        K: Tensor
            Kernel Gram matrix, of shape [batch, batch].
        d_K_Xi: Tensor
            Kernel gradients wrt. first input X. Shape: [batch, batch, dim]
        dK_dK_t: Tensor (Optional)
            Outer products of kernel gradients (used by SVN).
             Shape: [batch, batch, dim,  dim]
        """
        pass

class RBF(BaseKernel):
    """
        k(x, x') = exp( - || x - x'||**2 / (2 * ell**2))
    """
    def __init__(
        self,
        bandwidth=-1,
        analytic_grad=True,
        **kwargs,
    ):
        super().__init__(
            analytic_grad,
        )
        self.ell = bandwidth
        self.analytic_grad = analytic_grad

    def compute_bandwidth(
            self,
            X, Y
    ):
        """
            Older version.
        """

        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        pairwise_dists_sq = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        if self.ell < 0:  # use median trick
            try:
                h = torch.median(pairwise_dists_sq).detach()
            except Exception as e:
                print(pairwise_dists_sq)
                print(e)
        else:
            h = self.ell**2

        h = h / np.log(X.shape[0])

        # Clamp bandwidth
        tol = 1e-5
        if isinstance(h, torch.Tensor):
            h = torch.clamp(h, min=tol)
        else:
            h = np.clip(h, a_min=tol, a_max=None)

        return h, pairwise_dists_sq

    def eval(
            self,
            X, Y,
            M=None,
            compute_dK_dK_t=False,
            bw=None,
            **kwargs,
    ):

        assert X.shape == Y.shape

        if self.analytic_grad:
            if bw is None:
                h, pw_dists_sq = self.compute_bandwidth(X, Y)
            else:
                _, pw_dists_sq = self.compute_bandwidth(X, Y)
                h = bw

            K = (- pw_dists_sq / h).exp()
            d_K_Xi = K.unsqueeze(2) * (X.unsqueeze(1) - Y) * 2 / h
        else:
            raise NotImplementedError

        # Used for SVN updates
        dK_dK_t = None
        if compute_dK_dK_t:
            dK_dK_t = torch.einsum(
                    'bijk,bilm->bijm',
                    d_K_Xi.unsqueeze(3),
                    d_K_Xi.unsqueeze(2),
                )
        return (
            K,
            d_K_Xi,
            dK_dK_t
        )

class IMQ(BaseKernel):
    """
        IMQ Matrix-valued kernel, with metric M.
        k(x, x') = M^-1 (alpha + (x - y) M (x - y)^T ) ** beta
    """
    def __init__(
        self,
        # alpha=1,
        # beta=-0.5,
        alpha=1,
        beta=-0.5,
        hessian_scale=1,
        analytic_grad=True,
        median_heuristic=True,
        **kwargs,
    ):

        self.alpha = alpha
        self.beta = beta

        super().__init__(
            analytic_grad,
        )
        self.hessian_scale = hessian_scale
        self.median_heuristic = median_heuristic

    def eval(
        self,
        X, Y,
        M=None,
        compute_dK_dK_t=False,
        **kwargs,
        ):

        assert X.shape == Y.shape
        b, dim = X.shape

        # Empirical average of Hessian / Fisher matrices
        M = M.mean(dim=0)

        # PSD stabilization
        # M = 0.5 * (M + M.T)

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

        # Clamp bandwidth
        tol = 1e-5
        if isinstance(h, torch.Tensor):
            h = torch.clamp(h, min=tol)
        else:
            h = np.clip(h, a_min=tol, a_max=None)

        K = ( self.alpha + pairwise_dists_sq) ** self.beta
        d_K_Xi = self.beta * ((self.alpha + pairwise_dists_sq) ** (self.beta - 1)).unsqueeze(2) \
                 * ( -1 * (X.unsqueeze(1) - Y) @ M ) * 2 / h
                 # * ( (X.unsqueeze(1) - Y) @ M ) * 2 / h

        # Used for SVN updates
        dK_dK_t = None
        if compute_dK_dK_t:
            dK_dK_t = torch.einsum(
                    'bijk,bilm->bijm',
                    d_K_Xi.unsqueeze(3),
                    d_K_Xi.unsqueeze(2),
                )
        return (
            K,
            d_K_Xi,
            dK_dK_t
        )

class RBF_Anisotropic(RBF):
    """
        k(x, x') = exp( - (x - y) M (x - y)^T / (2 * d))
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
        )
        self.hessian_scale = hessian_scale
        self.median_heuristic = median_heuristic

    def eval(
        self,
        X, Y,
        M=None,
        compute_dK_dK_t=False,
        bw=None,
        **kwargs,
    ):

        assert X.shape == Y.shape

        # Empirical average of Hessian / Fisher matrices
        M = M.mean(dim=0)

        # PSD stabilization
        # M = 0.5 * (M + M.T)

        M *= self.hessian_scale

        X_M_Xt = X @ M @ X.t()
        X_M_Yt = X @ M @ Y.t()
        Y_M_Yt = Y @ M @ Y.t()

        if self.analytic_grad:
            if self.median_heuristic:
                bandwidth, pairwise_dists_sq = self.compute_bandwidth(X, Y)
            else:
                # bandwidth = self.hessian_scale * X.shape[1]
                bandwidth = self.hessian_scale
                pairwise_dists_sq = -2 * X_M_Yt + X_M_Xt.diag().unsqueeze(1) + Y_M_Yt.diag().unsqueeze(0)

            if bw is not None:
                bandwidth = bw

            K = (- pairwise_dists_sq / bandwidth).exp()
            d_K_Xi = K.unsqueeze(2) * ( (X.unsqueeze(1) - Y) @ M ) * 2 / bandwidth
        else:
            raise NotImplementedError

        # Used for SVN updates
        dK_dK_t = None
        if compute_dK_dK_t:
            dK_dK_t = torch.einsum(
                    'bijk,bilm->bijm',
                    d_K_Xi.unsqueeze(3),
                    d_K_Xi.unsqueeze(2),
                )
        return (
            K,
            d_K_Xi,
            dK_dK_t
        )

class Linear(BaseKernel):
    """
        k(x, x') = x^T x' + 1
    """
    def __init__(
        self,
        analytic_grad=True,
        subtract_mean=True,
        with_scaling=False,
        **kwargs,
    ):
        super().__init__(
            analytic_grad,
        )
        self.analytic_grad = analytic_grad
        self.subtract_mean = subtract_mean
        self.with_scaling = with_scaling

    def eval(
            self,
            X, Y,
            M=None,
            compute_dK_dK_t=False,
            **kwargs,
    ):

        assert X.shape == Y.shape
        batch, dim = X.shape

        if self.subtract_mean:
            mean = X.mean(0)
            X = X - mean
            Y = Y - mean

        if self.analytic_grad:
            K = X @ Y.t() + 1
            d_K_Xi = Y.repeat(batch, 1, 1)
        else:
            raise NotImplementedError

        if self.with_scaling:
            K = K / (dim + 1)
            d_K_Xi = d_K_Xi / (dim + 1)

        # Used for SVN updates
        dK_dK_t = None
        if compute_dK_dK_t:
            dK_dK_t = torch.einsum(
                    'bijk,bilm->bijm',
                    d_K_Xi.unsqueeze(3),
                    d_K_Xi.unsqueeze(2),
                )

        return (
            K,
            d_K_Xi,
            dK_dK_t
        )
