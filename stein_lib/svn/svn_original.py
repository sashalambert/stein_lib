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

from time import time
torch.set_default_tensor_type(torch.DoubleTensor)


class SVN_original:
    """
        SVN for debugging.
        Adapted from https://github.com/gianlucadetommaso/Stein-variational-samplers.
    """
    def __init__(
            self,
            model,
            particles,
            iters=100,
            eps=1.0
    ):
        self.model = model
        self.DoF = model.dim
        self.nParticles = particles.shape[1]
        self.nIterations = iters
        self.stepsize = eps
        self.particles = particles

    def apply(self):
        maxmaxshiftold = np.inf
        maxshift = np.zeros(self.nParticles)
        Q = np.zeros( (self.DoF, self.nParticles) )
        particle_history = []
        particle_history.append(np.copy(self.particles))

        # Time stats
        dts = []
        for iter_ in range(self.nIterations):

            particles_tn = torch.from_numpy(self.particles)
            F_tn = self.model.forward_model(particles_tn)
            J_tn = self.model.jacob_forward(particles_tn)
            gmlpt_tn = self.model.grad_log_p(particles_tn, F_tn, J_tn)
            Hmlpt_tn = self.model.GN_hessian(particles_tn, J_tn)

            gmlpt = gmlpt_tn.cpu().numpy()
            Hmlpt = Hmlpt_tn.cpu().numpy()
            M = np.mean(Hmlpt, 2)

            t_start = time()
            for i_ in range(self.nParticles):

                sign_diff = self.particles[:, i_, np.newaxis] - self.particles
                Msd = np.matmul(M, sign_diff)
                kern = np.exp( - 0.5 * np.sum( sign_diff * Msd, 0))
                gkern = Msd * kern

                mgJ = np.mean(- gmlpt * kern + gkern, 1)
                HJ  = np.mean(Hmlpt * kern ** 2, 2) + np.matmul(gkern, gkern.T) / self.nParticles

                Q[:, i_] = np.linalg.solve(HJ, mgJ)

                maxshift[i_] = np.linalg.norm(Q[:, i_], np.inf)

            self.particles += self.stepsize * Q
            maxmaxshift = np.max(maxshift)

            dt = time() - t_start
            print('dt (SVN): {}'.format(dt))
            dts.append(dt)

            if np.isnan(maxmaxshift) or (maxmaxshift > 1e20):
                print('Reset particles...')
                self.resetParticles()
                self.stepsize = 1
            elif maxmaxshift < maxmaxshiftold:
                self.stepsize *= 1.01
            else:
                self.stepsize *= 0.9
            maxmaxshiftold = maxmaxshift
            particle_history.append(np.copy(self.particles))

        dt_stats = np.array(dts)
        print("\nAvg. SVN_orignal compute time: {}".format(dt_stats.mean()))
        print("Std. dev. SVN_original compute time: {}\n".format(dt_stats.std()))

        return particle_history

    def resetParticles(self):
        self.particles = np.random.normal(scale=1, size=(self.DoF, self.nParticles) )
