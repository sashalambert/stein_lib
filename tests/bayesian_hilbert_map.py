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
from torch.distributions import Normal
from stein_lib.svgd.svgd import SVGD
from pathlib import Path
from stein_lib.models.bhm import BayesianHilbertMap
from tests.utils import create_movie_2D

torch.set_default_tensor_type(torch.DoubleTensor)

###### Params ######
num_particles = 100
iters = 3000
# iters = 200

# Sample intial particles
torch.manual_seed(1)

## Large Gaussian in center of intel map.
# prior_dist = Normal(loc=torch.tensor([3.,-10.]),
#                     scale=torch.tensor([10., 10.]))

## Small gaussian in corner of intel map.
prior_dist = Normal(loc=torch.tensor([12.,-3.]),
                    scale=torch.tensor([1.,1.]))

particles_0 = prior_dist.sample((num_particles,))

# Load model
import bhmlib
bhm_path = Path(bhmlib.__path__[0]).resolve()
model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter100.pt'
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

(particles,
 p_hist,
 pw_dists_sq) = svgd.apply(
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

create_movie_2D(
    p_hist,
    model.log_prob,
    to_numpy=True,
    save_path='./svgd_{}_bhm_intel_np_{}_eps_{}.mp4'.format(
        kernel_base_type,
        num_particles,
        step_size,
    ),
    ax_limits=ax_limits,
    opt='SVGD',
    kernel_base_type=kernel_base_type,
    num_particles=num_particles,
    eps=step_size,
)
