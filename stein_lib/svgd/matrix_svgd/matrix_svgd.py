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

from stein_lib.svgd.svgd import SVGD
from svgd.matrix_svgd.base_kernels import (
    RBF_Matrix,
    IMQ_Matrix,
    RBF_Weighted_Matrix,
)

# import sys
# np.set_printoptions(threshold=sys.maxsize)

class MatrixSVGD(SVGD):

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
        if self.kernel_base_type == 'IMQ_Matrix':
            return IMQ_Matrix(
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

        if self.kernel_structure is None:
            return self.base_kernel
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

        k_XX, grad_k = self.evaluate_kernel(X, M)

        b, dim = dlog_p.shape
        dlog_p = dlog_p.reshape(1, b, dim, 1)

        gradient = (k_XX.detach() @ dlog_p).mean(1).squeeze(-1)
        repulsive = grad_k.mean(1)
        return gradient, repulsive

    def evaluate_kernel(self, X, M=None):
        """
        Parameters
        ----------
        X :  tensor. Stein particles, of shape [batch, dim],
        M : (Optional) Negative Hessian or Fisher matrices. Tensor of shape [batch, dim, dim]

        Returns
        -------
        k_XX : tensor of shape [batch, batch, dim, dim]
        grad_k : tensor of shape [batch, batch, dim]

        """
        k_XX, grad_k = self.kernel.eval(
            X, X.clone().detach(),
            M,
        )
        return k_XX, grad_k
