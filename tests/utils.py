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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

#Animation function
def create_movie_2D(
        particle_hist,
        log_prob,
        save_path="/tmp/stein_movie.mp4",
        ax_limits=(-4, 4),
        to_numpy=False,
        kernel_base_type=None,
        opt=None,
        num_particles=None,
        eps=None,
):

    k_type = kernel_base_type,
    if kernel_base_type == 'RBF_Anisotropic':
        k_type = 'RBF_H'

    case_name = '{}-{} (np = {}, eps = {})'.format(
        opt,
        k_type,
        num_particles,
        eps,
    )

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.set_title(case_name + '\n' + str(0) + '$ ^{th}$ iteration')

    ngrid = 100
    x = np.linspace(ax_limits[0], ax_limits[1], ngrid)
    y = np.linspace(ax_limits[0], ax_limits[1], ngrid)
    X, Y = np.meshgrid(x,y)

    grid = np.vstack(
                (np.ndarray.flatten(X), np.ndarray.flatten(Y)),
            )
    if to_numpy:
        grid = torch.from_numpy(grid)
        z = log_prob(grid.t()).cpu().numpy()
        Z = np.exp(z).reshape(ngrid, ngrid)
    else:
        Z = np.exp(
            log_prob(grid),
        ).reshape(ngrid, ngrid)

    plt.contourf(X, Y, Z, 10)
    xlim = ax_limits
    ylim= ax_limits
    p_start = particle_hist[0]
    particles = plt.plot(p_start[:, 0], p_start[:, 1], 'ro', markersize=3)
    n_iter = len(particle_hist)

    def _init():  # only required for blitting to give a clean slate.
        # ax.set_title(str(0) + '$ ^{th}$ iteration')
        ax.set_title(case_name + '\n' + str(0) + '$ ^{th}$ iteration')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return particles

    def _animate(i):
        # ax.set_title(str(i) + '$ ^{th}$ iteration')
        ax.set_title(case_name + '\n' + str(i) + '$ ^{th}$ iteration')
        pos = particle_hist[i]
        particles[0].set_xdata(pos[:, 0])
        particles[0].set_ydata(pos[:, 1])
        return particles

    ani = animation.FuncAnimation(
        fig,
        _animate,
        init_func=_init,
        # interval=250,
        interval=100,
        blit=True,
        save_count=n_iter,
    )

    ani.save(save_path)
    plt.show()