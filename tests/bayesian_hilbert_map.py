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

torch.set_default_tensor_type(torch.DoubleTensor)

###### Params ######
# num_particles = 100
num_particles = 250
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
model_file = '/tmp/bhm_intel_res0.25_iter100.pt'
ax_limits = [[-10, 20],[-25, 5]]
model = BayesianHilbertMap(model_file, ax_limits)

#================== SVGD ===========================
particles = particles_0.clone().cpu().numpy()
particles = torch.from_numpy(particles)

# kernel_base_type = 'RBF'
# # optimizer_type = 'SGD'
# optimizer_type = 'Adam'
# step_size = 1.
# svgd = SVGD(
#     kernel_base_type=kernel_base_type,
#     kernel_structure=None,
#     median_heuristic=False,
#     repulsive_scaling=1.,
#     geom_metric_type=None,
#     verbose=True,
#     bandwidth=5.,
# )

kernel_base_type = 'RBF_Anisotropic'
# optimizer_type = 'SGD'
optimizer_type = 'Adam'
step_size = 0.25
# step_size = 0.
svgd = SVGD(
    kernel_base_type=kernel_base_type,
    kernel_structure=None,
    median_heuristic=False,
    repulsive_scaling=1.,
    geom_metric_type='fisher',
    verbose=True,
    bandwidth=5.,
)


# kernel_base_type = 'RBF_Anisotropic'
# optimizer_type = 'LBFGS' # 'FullBatchLBFGS'
# step_size = 0.1
# svgd = SVGD(
#     kernel_base_type=kernel_base_type,
#     kernel_structure=None,
#     median_heuristic=False,
#     repulsive_scaling=1.,
#     geom_metric_type='fisher',
#     verbose=True,
#     bandwidth=5.,
# )

## Optimize
(particles,
 p_hist,
 pw_dists,
 pw_dists_scaled) = svgd.apply(
    particles,
    model,
    iters,
    step_size,
    # use_analytic_grads=True,
    use_analytic_grads=False,
    optimizer_type=optimizer_type,
)

print("\nMean Est.: ", particles.mean(0))
print("Std Est.: ", particles.std(0))

#=============================================

# Construct Graph
(nodes,
 edge_lengths,
 edge_vals,
 edge_coll_num_pts,
 edge_coll_pts,
 params) = get_graph(
    particles.detach(),
    pw_dists,
    model,
    collision_thresh=5.,
    collision_res=0.25,
    connect_radius=5.,
    include_coll_pts=True,  # For debugging, visualization
)

# Plot Graph
plot_graph_2D(
    particles.detach(),
    nodes,
    model.log_prob,
    edge_vals=edge_vals,
    edge_coll_thresh=50.,
    # edge_coll_pts=edge_coll_pts,
    ax_limits=ax_limits,
    to_numpy=True,
    save_path='./graph_svgd_{}_bhm_intel_np_{}_eps_{}.png'.format(
        kernel_base_type,
        num_particles,
        step_size,
    ),
)

# # Make movie
# create_movie_2D(
#     p_hist,
#     model.log_prob,
#     to_numpy=True,
#     save_path='./svgd_{}_bhm_intel_np_{}_eps_{}.mp4'.format(
#         kernel_base_type,
#         num_particles,
#         step_size,
#     ),
#     ax_limits=ax_limits,
#     opt='SVGD',
#     kernel_base_type=kernel_base_type,
#     num_particles=num_particles,
#     eps=step_size,
# )
