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
from .base_kernels import RBF_Matrix
from abc import ABC, abstractmethod


class CompositeMatrixKernel(ABC):

    def __init__(
        self,
        kernel=RBF_Matrix,
        ctrl_dim=1,
        indep_controls=True,
        compute_hess_terms=False,
        **kargs,
    ):

        self.ctrl_dim = ctrl_dim
        self.base_kernel = kernel
        self.indep_controls = indep_controls
        self.compute_hess_terms = compute_hess_terms

    @abstractmethod
    def eval_svg_terms(self, X, Y, dlog_p, M=None, **kwargs):
        """
        Evaluates svgd gradient and repulsive terms.

        Parameters
        ----------
        X : tensor of shape [batch, dim]
        Y : tensor of shape [batch, dim]
        dlog_p : tensor of shape [batch, dim]
        M : tensor of shape [batch, dim, dim]
        kwargs :

        Returns
        -------
        kernel_Xj_Xi: Tensor
            Kernel Grammian, of shape [batch, batch, h, cl_size * dim, cl_size * dim]
            where cl_size is the size of the clique (ex.
        d_kernel_Xi: tensor of shape [batch, batch, dim]
        """
        pass


class matrix_iid_mp(CompositeMatrixKernel):

    def eval_svg_terms(
            self,
            X, Y,
            dlog_p,
            M=None,
            **kwargs,
    ):

        X = X.view(X.shape[0], -1, self.ctrl_dim)
        Y = Y.view(Y.shape[0], -1, self.ctrl_dim)

        # m: batch, h: horizon, d: ctrl_dim
        m, h, d = X.shape

        kernel_Xj_Xi = torch.zeros(m, m, h, d, h, d)

        d_kernel_Xi = torch.zeros(m, m, h, d)

        if M is not None:
            M = M.view(m, h, d, h, d)

        if self.indep_controls:
            for i in range(h):
                for q in range(self.ctrl_dim):
                    M_ii = None
                    if M is not None:
                        M_ii = M[:, i, q, i, q]
                    k_tmp, dk_tmp = self.base_kernel.eval(
                        X[:, i, q].reshape(-1, 1),
                        Y[:, i, q].reshape(-1, 1),
                        M_ii.reshape(-1, 1, 1),
                    )
                    kernel_Xj_Xi[:, :, i, q, i, q] += k_tmp.reshape(m, m)
                    d_kernel_Xi[:, :, i, q] += dk_tmp.squeeze(2)
        else:
            for i in range(h):
                M_ii = None
                if M is not None:
                    M_ii = M[:, i, :, i, :]
                k_tmp, dk_tmp = self.base_kernel.eval(
                    X[:, i, :],
                    Y[:, i, :],
                    M_ii,
                )
                kernel_Xj_Xi[:, :, i, :, i, :] += k_tmp
                d_kernel_Xi[:, :, i, :] += dk_tmp

        kernel_Xj_Xi = kernel_Xj_Xi.reshape(m, m, h * d, h * d)
        d_kernel_Xi = d_kernel_Xi.reshape(m, m, h * d)

        dlog_p = dlog_p.reshape(1, m, h * d, 1)
        k_Xj_Xi_dlog_p = kernel_Xj_Xi.detach() @ dlog_p

        return k_Xj_Xi_dlog_p, d_kernel_Xi


class matrix_first_order_mp(CompositeMatrixKernel):
    def eval_svg_terms(
            self,
            X, Y,
            dlog_p,
            M=None,
            **kwargs,
    ):

        X = X.view(X.shape[0], -1, self.ctrl_dim)
        Y = Y.view(Y.shape[0], -1, self.ctrl_dim)

        # m: batch, h: horizon, d: ctrl_dim
        m, h, d = X.shape

        k_Xj_Xi_dlog_p = torch.zeros(m, m, h, d)
        d_kernel_Xi = torch.zeros(m, m, h, d)
        dlog_p = dlog_p.reshape(1, m, h, d)

        if M is not None:
            M = M.view(m, h, d, h, d)

        for i in range(h):

            # clique : i-th node + Markov blanket
            if i == 0:
                clique = [i, i + 1]
                cl_i = 0
            elif i == h-1:
                clique = [i - 1, i]
                cl_i = 1
            else:
                clique = [i - 1, i, i + 1]
                cl_i = 1
            num = len(clique)

            if self.indep_controls:
                for q in range(self.ctrl_dim):
                    M_i = None
                    if M is not None:
                        M_i = M[:, clique, q, :, q][:, :, clique]
                        M_i = M_i.reshape(-1, num, num)
                    k_tmp, dk_tmp = self.base_kernel.eval(
                        X[:, clique, q].reshape(-1, num),
                        Y[:, clique, q].reshape(-1, num),
                        M_i,
                    )
                    k_dlog_p = (k_tmp @ dlog_p[:, :, clique, q].unsqueeze(-1)).squeeze(-1)
                    k_Xj_Xi_dlog_p[:, :, i, q] += k_dlog_p[:, :, cl_i]
                    d_kernel_Xi[:, :, i, q] += dk_tmp[:, :, cl_i]
            else:
                M_i = None
                if M is not None:
                    M_i = M[:, clique, :, :, :][:, :, :, clique, :]
                    M_i = M_i.reshape(-1, num * d, num * d)
                k_tmp, dk_tmp = self.base_kernel.eval(
                    X[:, clique, :].reshape(-1, num * d),
                    Y[:, clique, :].reshape(-1, num * d),
                    M_i,
                )
                dlog_p_cl = dlog_p[:, :, clique, :].reshape(1, m, num * d, 1)
                k_dlog_p = (k_tmp @ dlog_p_cl).reshape(m, m, num, d)
                dk_tmp = dk_tmp.reshape(m, m, num, d)
                k_Xj_Xi_dlog_p[:, :, i, :] += k_dlog_p[:, :, cl_i, :]
                d_kernel_Xi[:, :, i, :] += dk_tmp[:, :, cl_i, :]

        k_Xj_Xi_dlog_p = k_Xj_Xi_dlog_p.reshape(m, m, h * d)
        d_kernel_Xi = d_kernel_Xi.reshape(m, m, h * d)

        return k_Xj_Xi_dlog_p, d_kernel_Xi

