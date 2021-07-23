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
from stein_lib.svgd.svgd import SVGD
from time import time


class SVN(SVGD):

    def __init__(
            self,
            kernel_base_type='RBF',
            kernel_structure=None,
            verbose=False,
            control_dim=None,
            repulsive_scaling=1.,
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
            Kernel Grammian. Shape: [num_particles, num_particles]
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
        k_sq = (k_XX ** 2).unsqueeze(-1).unsqueeze(-1) # b x b x 1 x 1
        H_ii = - Hess * k_sq + dk_dk_t
        H = H_ii.mean(dim=1)
        return H

    def get_svn_terms(
        self,
        X,
        dlog_p,
        dlog_lh,
        Hess,
        Hess_prior=None,
        transpose=False,
    ):
        """
        Parameters
        ----------
        X : (Tensor)
            Stein particles, of shape [num_particles, dim]
            or of shape [dim, num_particles]. If Tensor dimension is greater than 2,
            extra dimensions will be flattened.
        dlog_p : (Tensor)
            Score function, of shape [num_particles, dim]
            or of shape [dim, num_particles].  If Tensor dimension is greater than 2,
            extra dimensions will be flattened.
        Hess : (Tensor)
            Hessian of prob. density, of shape [num_particles, dim, dim]
            or of shape [dim, dim, num_particles].  If Tensor dimension is greater than 3,
            it will be reshaped appropriately such that dimension is 3.
        trans

        pose: Bool
            Transpose input and output Tensors.
        Returns
        -------
        Phi: (Tensor)
            Empirical Stein gradient, of shape [num_particles, dim]
        """

        shape_original = X.shape

        X, dlog_p, Hess = self.reshape_inputs(
            X,
            dlog_p,
            Hess,
            transpose,
        )

        if self.geom_metric_type is None:
            pass
        elif self.geom_metric_type == 'full_hessian':
            assert Hess is not None
            M = - Hess
            M += 1.e-6 * torch.eye(M.shape[1], M.shape[2])
        elif self.geom_metric_type == 'fisher':
            ## Average Fisher matrix (likelihood only)
            np = dlog_lh.shape[0]
            M = torch.bmm(dlog_lh.reshape(np, -1, 1,), dlog_lh.reshape(np, 1, -1))
            M += 1.e-6 * torch.eye(M.shape[1], M.shape[2])
        elif self.geom_metric_type == 'jacobian_product':
            ## Average Fisher matrix (full posterior gradient)
            dim = dlog_p.shape[-1]
            M = torch.bmm(dlog_p.view(-1, dim, 1,), dlog_p.view(-1, 1, dim))
            M += 1.e-3 * torch.eye(M.shape[1], M.shape[2])
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

        (k_XX,
         grad_k,
         dk_dk_t) = self.kernel.eval(
            X, X.clone().detach(),
            M,
            compute_dK_dK_t=True,
        )

        ## Phi - first variation ###
        grad, rep = self.get_svgd_terms(
            X,
            dlog_p,
            M,
        )
        # if True:
        if self.verbose:
            print('gradient l2-norm: {:5.4f}'.format(
                grad.norm().detach().cpu().numpy()))
            print('repulsive l2-norm: {:5.4f}'.format(
                rep.norm().detach().cpu().numpy()))

        phi = grad + self.repulsive_scaling * rep

        # phi += 0.005 * torch.randn(phi.shape)

        ## Q - Second varation ##
        H = self.get_second_variation(k_XX, dk_dk_t, Hess)
        H += 1.e-4 * torch.eye(H.shape[1], H.shape[2])

        Q = torch.solve(phi.unsqueeze(2), H).solution

        ### Debugging - use scipy solver
        # phi_np = phi.unsqueeze(2).clone().detach().numpy()
        # H_np = H.clone().detach().numpy()
        # Q_np = np.zeros_like(phi_np)
        # for i in range(X.shape[0]):
        #     Q_np[i] = scipy_solve(H_np[i], phi_np[i])
        # Q = torch.from_numpy(Q_np)

        Q = Q.squeeze()

        # Reshape Q to match original tensor dimensions
        if transpose:
            Q = Q.t()
        Q = Q.reshape(shape_original)

        return (
            Q,
            k_XX,
            grad_k,
        )

    def apply(
            self,
            X,
            model,
            iters=100,
            eps=1.,
            use_analytic_grads=False,
    ):
        """
        SVGD updates.

        Parameters
        ----------
        X : (Tensor) of nd.array
            Stein particles, of shape [dim, num_particles]
        eps : Float
            Step size.
        """

        particle_history = []
        particle_history.append(X.clone().cpu().numpy())

        X = torch.autograd.Variable(X, requires_grad=True)

        # Time stats
        dts = []
        for i in range(iters):

            if use_analytic_grads:
                F = model.forward_model(X)
                J = model.jacob_forward(X)
                dlog_p = model.grad_log_p(X, F, J)
                # Gauss-Newton approximation
                Hess = model.GN_hessian(X, J)
                Hess = -1 * Hess
            else:
                log_p = model.log_prob(X).unsqueeze(1)
                dlog_p = torch.autograd.grad(
                    log_p.sum(),
                    X,
                    create_graph=True,
                )[0]

                ## Full Hessian
                Hess = self.get_jacobian(dlog_p, X)

            t_start = time()

            (Q,
             k_XX,
             grad_k) = self.get_svn_terms(
                X,
                dlog_p,
                dlog_p.transpose(0,1),
                Hess,
                None,
                transpose=True
            )

            dt = time() - t_start
            print('dt (SVN): {}'.format(dt))
            dts.append(dt)

            X = X + eps * Q

            X = X.detach()
            X.requires_grad = True

            particle_history.append(X.clone().detach().cpu().numpy())

        dt_stats = np.array(dts)
        print("\nAvg. SVN compute time: {}".format(dt_stats.mean()))
        print("Std. dev. SVN compute time: {}\n".format(dt_stats.std()))

        return X, particle_history