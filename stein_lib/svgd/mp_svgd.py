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

from .svgd import SVGD
from .mp_composite_kernels import (
    first_order_mp,
    # ternary_mp,
    iid_mp,
)


class MP_SVGD(SVGD):
    """
        Message Passing SVGD.
    """

    def get_kernel(
            self,
            **kernel_params,
    ):

        if self.kernel_structure is None:
            return self.base_kernel
        elif self.kernel_structure == 'iid_mp':
            return iid_mp(
                ctrl_dim=self.ctrl_dim,
                kernel=self.base_kernel,
                **kernel_params,
          )
        elif self.kernel_structure == 'first_order_mp':
            return first_order_mp(
                ctrl_dim=self.ctrl_dim,
                kernel=self.base_kernel,
                **kernel_params,
            )
        else:
            raise IOError('Kernel structure not recognized for MP-SVGD: ',
                          self.kernel_structure,)

    def get_svgd_terms(
            self,
            X,
            dlog_p,
            M=None,
    ):
        """
        Handle Message-Passing SVGD update.

        Parameters
        ----------
        X :  Stein particles. Tensor of shape [batch, dim],
        dlog_p : tensor of shape [batch, dim]
        M : (Optional) Negative Hessian or Fisher matrices. Tensor of shape [batch, dim, dim]

        Returns
        -------
        gradient: tensor of shape [batch, dim]
        repulsive: tensor of shape [batch, dim]

        """

        k_XX, grad_k, pw_dists_sq = self.evaluate_kernel(X, M)

        gradient = (k_XX.detach() * dlog_p.unsqueeze(0)).mean(1)
        repulsive = grad_k.mean(1)

        return gradient, repulsive, pw_dists_sq

    def evaluate_kernel(self, X, M=None):
        """

        Parameters
        ----------
        X :  tensor. Stein particles, of shape [batch, dim],
        M : (Optional) Negative Hessian or Fisher matrices. Tensor of shape [batch, dim, dim]

        Returns
        -------
        k_XX : tensor of shape [batch, batch, dim]
        grad_k : tensor of shape [batch, batch, dim]

        """
        k_XX, grad_k, _, pw_dists_sq = self.kernel.eval(
            X, X.clone().detach(),
            M,
            compute_dK_dK_t=False,
        )
        return k_XX, grad_k, pw_dists_sq
