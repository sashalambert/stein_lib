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
iters = 200
eps_list = [0.1]

# Sample intial particles
torch.manual_seed(1)
prior_dist = Normal(loc=0., scale=1.)
particles_0 = prior_dist.sample((num_particles, 2))

# Load model
import bhmlib
bhm_path = Path(bhmlib.__path__[0]).resolve()
model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter010.pt'

model = BayesianHilbertMap(model_file)


for eps in eps_list:

    #================== SVGD ===========================

    particles = particles_0.clone().cpu().numpy()
    particles = torch.from_numpy(particles)
    # kernel_base_type = 'RBF_Anisotropic' # 'RBF', 'IMQ'
    kernel_base_type = 'RBF'
    # optimizer_type = 'LBFGS' # 'FullBatchLBFGS'
    optimizer_type = 'SGD'

    svgd = SVGD(
        kernel_base_type=kernel_base_type,
        kernel_structure=None,
        median_heuristic=False,
        repulsive_scaling=2.,
        geom_metric_type='fisher',
        verbose=True,
    )

    particles, p_hist, pw_dists_sq = svgd.apply(
                            particles,
                            model,
                            iters,
                            eps,
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
            eps,
        ),
        ax_limits=[-5, 5],
        opt='SVGD',
        kernel_base_type=kernel_base_type,
        num_particles=num_particles,
        eps=eps,
    )
