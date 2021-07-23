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
from svmpc_np.svgd.matrix_svgd.matrix_svgd import MatrixSVGD
from svmpc_np.svgd.base_kernels import (
    RBF_Anisotropic,
)
from svmpc_np.svgd.matrix_svgd.base_kernels import (
    RBF_Matrix,
    RBF_Weighted_Matrix,
)
from svmpc_np.svgd.matrix_svgd.mp_composite_kernels import (
    matrix_iid_mp,
    matrix_first_order_mp,
)


class MP_MatrixSVGD(MatrixSVGD):

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
        if self.kernel_base_type == 'RBF_Matrix':
            return RBF_Matrix(
                **kernel_params,
            )
        elif self.kernel_base_type == 'RBF_Weighted_Matrix':
            return RBF_Weighted_Matrix(
                **kernel_params,
            )
        else:
            raise IOError('Matrix-SVGD kernel type not recognized: ',
                          self.kernel_base_type)

    def get_kernel(
            self,
            **kernel_params,
    ):

        if self.kernel_structure == 'matrix_iid_mp':
            return matrix_iid_mp(
                ctrl_dim=self.ctrl_dim,
                kernel=self.base_kernel,
                **kernel_params,
          )
        elif self.kernel_structure == 'matrix_first_order_mp':
            return matrix_first_order_mp(
                ctrl_dim=self.ctrl_dim,
                kernel=self.base_kernel,
                **kernel_params,
            )
        else:
            raise IOError('Kernel structure not recognized for matrix-SVGD: ',
                          self.kernel_structure,)

    def get_svgd_terms(
            self,
            X,
            dlog_p,
            M=None,
    ):

        """
        Handle matrix-valued SVGD terms.

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

        k_dlog_p, grad_k = self.kernel.eval_svg_terms(
            X, X.clone().detach(),
            dlog_p,
            M,
        )

        gradient = k_dlog_p.mean(1).squeeze(-1)
        repulsive = grad_k.mean(1)

        return gradient, repulsive
