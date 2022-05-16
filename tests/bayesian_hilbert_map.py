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
from torch.distributions import Normal, Uniform
from stein_lib.models.gaussian_mixture import mixture_of_gaussians
from stein_lib.svgd.svgd import SVGD
from pathlib import Path
from stein_lib.models.bhm import BayesianHilbertMap
from stein_lib.utils import create_movie_2D, plot_graph_2D
from stein_lib.prm_utils import get_graph
from stein_lib.svgd.base_kernels import RBF, RBF_Anisotropic
from stein_lib.svgd.LBFGS import FullBatchLBFGS, LBFGS
torch.set_default_tensor_type(torch.DoubleTensor)

###### Params ######
# num_particles = 100
num_particles = 250
# num_particles = 1
# iters = 3000
# iters = 200
iters = 100
# iters = 1

# Sample intial particles
torch.manual_seed(1)

## Large Gaussian in center of intel map.
# prior_dist = Normal(loc=torch.tensor([3.,-10.]),
#                     scale=torch.tensor([10., 10.]))

## Small gaussian in corner of intel map.
# prior_dist = Normal(loc=torch.tensor([12.,-3.]),
#                     scale=torch.tensor([1.,1.]))

## Two small gaussians in opposing corners of intel map.
# sigma = 5.
# radii_list = [[sigma, sigma],] * 2
# prior_dist = mixture_of_gaussians(
#     num_comp=2,
#     mu_list=[[12.,-3.], [-5, -18] ],
#     sigma_list=radii_list,
# )

## Uniform distribution
prior_dist = Uniform(low=torch.tensor([-10., -25.]),
                    high=torch.tensor([20., 5.]))

particles_0 = prior_dist.sample((num_particles,))

# Load model
import bhmlib
bhm_path = Path(bhmlib.__path__[0]).resolve()
# model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter100.pt'
# model_file = '/tmp/bhm_intel_res0.25_iter100.pt'
model_file = '/tmp/bhm_intel_res0.25_iter900.pt'
ax_limits = [[-10, 20], [-25, 5]]
model = BayesianHilbertMap(model_file, ax_limits)


particles = particles_0.clone().cpu().numpy()
particles = torch.from_numpy(particles)


#================== Kernel ===========================

# kernel = RBF(
#     hessian_scale=1.0,
#     analytic_grad=True,
#     median_heuristic=False,
#     bandwidth=5.0,
# )

# kernel = RBF(
#     hessian_scale=1.0,
#     analytic_grad=True,
#     median_heuristic=False,
#     bandwidth=1.0,
# )

kernel = RBF_Anisotropic(
    hessian_scale=1.0,
    analytic_grad=True,
    median_heuristic=False,
    bandwidth=1.0,
)

#================== Optimizer ===========================

# optimizer = torch.optim.SGD([particles], lr=1.)

optimizer = torch.optim.Adam([particles], lr=0.25)

# optimizer = torch.optim.LBFGS(
#    [particles],
#    lr=0.1,
#    max_iter=100,
#    # max_eval=20 * 1.25,
#    tolerance_change=1e-9,
#    history_size=25,
#    line_search_fn=None, #'strong_wolfe'
# )

# optimizer = FullBatchLBFGS(
#    [particles],
#    lr=0.1,
#    history_size=25,
#    line_search='None', #'Wolfe'
# )

#================== SVGD ===========================

svgd = SVGD(
    kernel=kernel,
    kernel_structure=None,
    repulsive_scaling=1.,
    geom_metric_type='fisher',
    verbose=True,
)

## Optimize
(particles,
 p_hist,
 pw_dists,
 pw_dists_scaled) = svgd.apply(
    particles,
    model,
    iters,
    # use_analytic_grads=True,
    use_analytic_grads=False,
    optimizer=optimizer,
)

print("\nMean Est.: ", particles.mean(0))
print("Std Est.: ", particles.std(0))

#================== Graph ===========================

# (nodes,
#  edge_lengths,
#  edge_vals,
#  edge_coll_binary,
#  edge_coll_num_pts,
#  edge_coll_pts,
#  params) = get_graph(
#     particles.detach(),
#     pw_dists,
#     model,
#     collision_thresh=5.,
#     collision_res=0.25,
#     connect_radius=5.,
#     include_coll_pts=True,  # For debugging, visualization
# )

#================== Visualization ===========================

# Plot Graph
# plot_graph_2D(
#     particles.detach(),
#     nodes,
#     model.log_prob,
#     edge_vals=edge_vals,
#     edge_coll_thresh=50.,
#     # edge_coll_pts=edge_coll_pts,
#     ax_limits=ax_limits,
#     to_numpy=True,
#     save_path='./graph_svgd_{}_bhm_intel_np_{}.png'.format(
#         kernel.__class__.__name__,
#         num_particles,
#     ),
# )

# Make movie
create_movie_2D(
    p_hist,
    model.log_prob,
    to_numpy=True,
    save_path='./svgd_{}_bhm_intel_np_{}.mp4'.format(
        kernel.__class__.__name__,
        num_particles,
    ),
    ax_limits=ax_limits,
    opt='SVGD',
    kernel_base_type=kernel.__class__.__name__,
    num_particles=num_particles,
)
