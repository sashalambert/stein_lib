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
from .base_kernels import RBF
from abc import ABC, abstractmethod


class CompositeKernel(ABC):

    def __init__(
        self,
        kernel=RBF,
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
    def eval(self, X, Y, M=None, **kwargs):

        """

        Parameters
        ----------
        X : tensor of shape [batch, dim]
        Y : tensor of shape [batch, dim]
        M : tensor of shape [dim, dim]
        kwargs :

        Returns
        -------

        """
        pass


class iid(CompositeKernel):

    def eval(
            self,
            X, Y,
            M=None,
            compute_dK_dK_t=False,
            **kwargs,
    ):
        X = X.view(X.shape[0], -1, self.ctrl_dim)
        Y = Y.view(Y.shape[0], -1, self.ctrl_dim)

        # m: batch, h: horizon, d: ctrl_dim
        m, h, d = X.shape

        # Keep another batch-dim for grad. mult. later on.
        kernel_Xj_Xi = torch.zeros(m, m, h, d)

        # shape : (m, h, d)
        d_kernel_Xi = torch.zeros(m, m, h, d)

        # shape : (m, h, d)
        d_k_Xj_dk_Xj = torch.zeros(m, m, h, d, h, d)

        if M is not None:
            M = M.view(h, d, h, d).transpose(1, 2)  # shape: (h, h, d, d)

        if self.indep_controls:
            for i in range(h):
                for q in range(self.ctrl_dim):
                    M_ii = None
                    if M is not None:
                        M_ii = M[i, i, q, q]
                    k_tmp, dk_tmp, dk_dk_t_tmp = self.base_kernel.eval(
                        X[:, i, q].reshape(-1, 1),
                        Y[:, i, q].reshape(-1, 1),
                        M_ii.reshape(1, 1),
                        compute_dK_dK_t=compute_dK_dK_t,
                    )
                    kernel_Xj_Xi[:, :, i, q] += k_tmp
                    d_kernel_Xi[:, :, i, q] += dk_tmp.squeeze(2)
                    if compute_dK_dK_t:
                        d_k_Xj_dk_Xj[:, :, i, q, i, q] += dk_dk_t_tmp.view(m, m)
        else:
            for i in range(h):
                M_ii = None
                if M is not None:
                    M_ii = M[i, i, :, :]
                k_tmp, dk_tmp, dk_dk_t_tmp = self.base_kernel.eval(
                    X[:, i, :],
                    Y[:, i, :],
                    M_ii,
                    compute_dK_dK_t=compute_dK_dK_t,
                )
                kernel_Xj_Xi[:, :, i, :] += k_tmp.unsqueeze(2)
                d_kernel_Xi[:, :, i, :] += dk_tmp
                if compute_dK_dK_t:
                    d_k_Xj_dk_Xj[:, :, i, :, i, :] += dk_dk_t_tmp

        kernel_Xj_Xi = kernel_Xj_Xi.reshape(m, m, h * d).sum(2)

        d_kernel_Xi = d_kernel_Xi.reshape(m, m, h * d)
        d_k_Xj_dk_Xj = d_k_Xj_dk_Xj.reshape(m, m, h * d, h * d)

        return kernel_Xj_Xi, d_kernel_Xi, d_k_Xj_dk_Xj
