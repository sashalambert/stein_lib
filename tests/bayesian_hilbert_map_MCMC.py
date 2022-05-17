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
import numpy as np
from stein_lib.models.gaussian_mixture import mixture_of_gaussians
from stein_lib.mcmc.sgld import LangevinDynamics, MetropolisAdjustedLangevin
from pathlib import Path
from stein_lib.models.bhm import BayesianHilbertMap
from stein_lib.utils import create_movie_2D, plot_graph_2D
from stein_lib.prm_utils import get_graph
from stein_lib.svgd.base_kernels import RBF, RBF_Anisotropic
from stein_lib.svgd.LBFGS import FullBatchLBFGS, LBFGS
from tqdm import tqdm
from pyro.infer.mcmc import NUTS, MCMC
import pyro

if not torch.cuda.is_available():
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

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
prior_dist = Uniform(low=torch.tensor([-10., -25.]).to(device),
                    high=torch.tensor([20., 5.]).to(device))

particles_0 = prior_dist.sample((num_particles,))

# Load model
import bhmlib
bhm_path = Path(bhmlib.__path__[0]).resolve()
# model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter100.pt'
# model_file = '/tmp/bhm_intel_res0.25_iter100.pt'
model_file = '/tmp/bhm_intel_res0.25_iter900.pt'
ax_limits = [[-10, 20], [-25, 5]]
model = BayesianHilbertMap(model_file, ax_limits, device)


# particles = particles_0.clone().cpu().numpy()
# particles = torch.from_numpy(particles)
particles = particles_0.clone().detach()

#================== Optimizer ===========================

# optimizer = torch.optim.SGD([particles], lr=1.)

optimizer = torch.optim.Adam([particles], lr=0.25)

#================== SVGD ===========================

# svgd = SVGD(
#     kernel=kernel,
#     kernel_structure=None,
#     repulsive_scaling=1.,
#     geom_metric_type='fisher',
#     verbose=True,
# )
#
# ## Optimize
# (particles,
#  p_hist,
#  pw_dists,
#  pw_dists_scaled) = svgd.apply(
#     particles,
#     model,
#     iters,
#     # use_analytic_grads=True,
#     use_analytic_grads=False,
#     optimizer=optimizer,
# )

#================== SGLD ===========================

# x = torch.randn([5, 2], requires_grad=True)
x = particles
x.requires_grad = True
max_itr = int(500)
langevin_dynamics = LangevinDynamics(
    lr=0.1,
    lr_final=1e-2,
)
# langevin_dynamics = MetropolisAdjustedLangevin(
#     x,
#     model,
#     lr=0.1,
#     lr_final=1e-2,
#     max_itr=max_itr,
# )

particles, p_hist = langevin_dynamics.apply(x, model, max_itr)

#================== HMC ===========================

# #TODO: modify BHM model to match
# def pyro_model(data):
#     y = pyro.sample('y', model(), obs=data)
#     return y
#
# nuts_kernel = NUTS(model.log_prob, adapt_step_size=True)
# mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=300)
# mcmc.run(particles)
# particles = mcmc.get_samples()
# particles = particles.unsqueeze()

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
    model,
    to_numpy=True,
    save_path='./sgld_bhm_intel_np_{}.mp4'.format(
        num_particles,
    ),
    ax_limits=ax_limits,
    opt='SGLD',
    kernel_base_type=None,
    num_particles=num_particles,
)