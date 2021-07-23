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

from svgd.mp_svgd import MP_SVGD
from stein_lib.svn import SVN


class MP_SVN(SVN, MP_SVGD):

    def __init__(
            self,
            kernel_base_type='RBF',
            kernel_structure=None,
            verbose=False,
            control_dim=None,
            repulsive_scaling=1,
            **kernel_params,
    ):

        super().__init__(
            kernel_base_type,
            kernel_structure,
            verbose,
            control_dim,
            repulsive_scaling,
            **kernel_params,
        )

    def get_second_variation(
            self,
            k_XX,
            dk_dk_t,
            Hess,
    ):
        """

        Parameters
        ----------
        k_XX : tensor
            Kernel Grammian. Shape: [num_particles, num_particles, dim]
        dk_dk_t : tensor
            Outer products of kernel gradients.
            Shape: [num_particles, num_particles, dim, dim]
        Hess : tensor
            Hessian of log_prob.
            Shape: [num_particles, dim, dim]

        Returns
        -------
        H : tensor
            Second variation. Shape [num_particles, dim, dim].
        """
        k_sq = (k_XX ** 2).unsqueeze(-1) # b x b x d x 1
        H_ii = - Hess * k_sq + dk_dk_t
        H = H_ii.mean(dim=1)
        return H

    def get_svgd_terms(
            self,
            X,
            dlog_p,
            M=None,
    ):

        # Use message-passing format
        return MP_SVGD.get_svgd_terms(
                self, X, dlog_p, M,
            )
