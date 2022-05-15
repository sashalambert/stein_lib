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
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation


def get_jacobian(
        gradient,
        X,
):
    """
    Returns the Jacobian matrix, given the gradient
    Parameters
    ----------
    gradient : (Tensor)
        Of shape [dim, batch]
    X : (Tensor)
        of shape [dim, batch]
    Returns
    -------
    J : (Tensor)
        Jacobian, of shape [dim, dim, batch]
    """
    dg_dXi = [
        torch.autograd.grad(
            gradient[:, i].sum(),
            X,
            retain_graph=True,
        )[0] for i in range(gradient.shape[1])
    ]
    J = torch.stack(dg_dXi, dim=1)
    return J


def calc_pw_distances(X):
    """
    Returns the pairwise distances between particles.
    Parameters
    ----------
    X : Tensor
        Points. of shape [dim, batch]
    """
    XX = X.matmul(X.t())
    pairwise_dists_sq = -2 * XX + XX.diag().unsqueeze(1) + XX.diag().unsqueeze(0)
    pw_dists = torch.sqrt(pairwise_dists_sq)
    return pw_dists


def calc_scaled_pw_distances(X, M):
    """
    Returns the metric-scaled / anisotropic pairwise distances between particles.
    Parameters
    ----------
    X : Tensor
        Points. of shape [dim, batch]
    M : Tensor
        Metric. of shape [dim, dim]
    """
    X_M_Xt = X @ M @ X.t()
    pw_dists_sq = -2 * X_M_Xt + X_M_Xt.diag().unsqueeze(1) + X_M_Xt.diag().unsqueeze(0)
    pw_dists = torch.sqrt(pw_dists_sq)
    return pw_dists


def plot_graph_2D(
        particles,
        edges,
        log_prob,
        edge_vals=None,
        edge_coll_pts=None,
        edge_coll_thresh=None,
        save_path='/tmp/graph.png',
        to_numpy=False,
        ax_limits=[[-4, 4],[4, 4]],
):

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()

    ngrid = 100
    x = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
    X, Y = np.meshgrid(x,y)

    grid = np.vstack(
                (np.ndarray.flatten(X), np.ndarray.flatten(Y)),
            )

    if to_numpy:
        grid = torch.from_numpy(grid)
        z = log_prob(grid.t()).cpu().numpy()
        Z = np.exp(z).reshape(ngrid, ngrid)
        particles = particles.detach().cpu().numpy()
    else:
        Z = np.exp(
            log_prob(grid),
        ).reshape(ngrid, ngrid)

    plt.contourf(X, Y, Z, 10)

    for i in range(edges.shape[0]):
        node_pair = edges[i]
        color = 'k'
        if edge_vals is not None:
            if edge_vals[i] < edge_coll_thresh:
                color = 'b'
        plt.plot(
            particles[node_pair, 0],
            particles[node_pair, 1],
             color,
             markersize=1,
        )

    if edge_coll_pts is not None:
        plt.plot(edge_coll_pts[:, 0], edge_coll_pts[:, 1], 'go', markersize=2)

    xlim = ax_limits[0]
    ylim = ax_limits[1]
    plt.plot(particles[:, 0], particles[:, 1], 'ro', markersize=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(save_path)
    plt.show()


def create_movie_2D(
        particle_hist,
        log_prob,
        save_path="/tmp/stein_movie.mp4",
        ax_limits=[[-4, 4],[4, 4]],
        to_numpy=False,
        kernel_base_type=None,
        opt=None,
        num_particles=None,
):

    k_type = kernel_base_type,
    if kernel_base_type == 'RBF_Anisotropic':
        k_type = 'RBF_H'

    # case_name = '{}-{} (np = {}, eps = {})'.format(
    case_name = '{}-{} (np = {})'.format(
        opt,
        k_type,
        num_particles,
    )

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.set_title(case_name + '\n' + str(0) + '$ ^{th}$ iteration')

    ngrid = 100
    x = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
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
    xlim = ax_limits[0]
    ylim = ax_limits[1]
    # xlim = [ax_limits[0][0] - 10, ax_limits[0][1] + 10]
    # ylim = [ax_limits[1][0] - 10, ax_limits[1][1] + 10]
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
        frames=n_iter,
        init_func=_init,
        # interval=250,
        interval=100,
        save_count=n_iter,
    )

    ani.save(save_path)
    plt.show()
