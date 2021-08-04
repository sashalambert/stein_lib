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

from stein_lib.models.gaussian_mixture import mixture_of_gaussians
from tests.utils import create_movie_2D

torch.set_default_tensor_type(torch.DoubleTensor)

###### Params ######
num_particles = 100
iters = 200
eps_list = [0.1]

# Sample intial particles
torch.manual_seed(1)
# prior_dist = Normal(loc=0., scale=1.)
# prior_dist = Normal(loc=0., scale=0.5)
prior_dist = Normal(loc=-4, scale=0.5)
particles_0 = prior_dist.sample((2, num_particles))

# Load model

#### Bi-modal Gaussian mixture #########
# sigma = 1.
# nc = 2
# centers_list=[
#     [-1.5, 0.],
#     [1.5, 0.],
# ]

#### Toy Bayesian occupancy map ###########
sigma = 0.4
nc = 37 # number of components
s = 2. # rbf spacing
centers_list=[
    [-2*s, 2*s] ,  [-s, 2*s],  [0., 2*s],  [s, 2*s], [2*s, 2*s],
    [-2*s, s] , [-s, s],  [0., s],  [s, s], [2*s, s],
    [-2*s, 0.]  , [-s, 0.], [0., 0.], [s, 0.], [2*s, 0.],
     [-s, -s], [2*s, -s],
    [-2*s, -2*s], [-s, -2*s], [0., -2*s],  [2*s, -2*s],

    [-2*s, 1.5*s] ,  [-2*s, 0.5*s],  [-1.5*s, 0],  [-0.5*s, 0],
    [0.5*s, 0], [1.5*s, 0], [2*s, -0.5*s], [2*s, -1.5*s],
    [-1*s, -0.5*s], [-1*s, -1.5*s], [-0.5*s, -2*s], [0.5*s, -2*s],
    [-1.5*s, -2*s], [1*s, 0.5*s],  [1*s, 1.5*s], [1.5*s, 2*s],
]
############

radii_list = [
    [sigma, sigma],
] * nc

model = mixture_of_gaussians(
    num_comp=nc,
    mu_list=centers_list,
    sigma_list=radii_list,
)


for eps in eps_list:

    ### SVN  ########
    # particles = particles_0.clone().cpu().numpy()
    # particles = torch.from_numpy(particles)
    #
    # eps = 1.
    # # kernel_base_type='RBF_Anisotropic'
    # # kernel_base_type='RBF'
    # kernel_base_type='IMQ'
    # svn = SVN(
    #     kernel_base_type=kernel_base_type,
    #     compute_hess_terms=True,
    #     use_hessian_metric=False,
    #     umedian_heurstic=True,
    #     # geom_metric_type='fisher',
    #     geom_metric_type='jacobian_product',
    #     # geom_metric_type='full_hessian',
    #     )
    #
    # particles, p_hist = svn.apply(
    #                         particles,
    #                         model,
    #                         iters,
    #                         eps,
    #                         optimizer_type='LBFGS'
    #                     )
    #
    # create_movie_2D(
    #     p_hist,
    #     model.log_prob,
    #     ax_limits=(-4, 4),
    #     to_numpy=True,
    #     save_path='./svn_tests/svn_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #     # save_path='./svn_tests/svn_{}_gaussian_mix_hard_np_{}_eps_{}.mp4'.format(
    #         kernel_base_type,
    #         num_particles,
    #         eps,
    #     ),
    #     opt='SVN',
    #     kernel_base_type=kernel_base_type,
    #     num_particles=num_particles,
    #     eps=eps,
    # )

    #================== SVGD ===========================

    particles = particles_0.clone().cpu().numpy()
    particles = torch.from_numpy(particles)
    kernel_base_type = 'RBF_Anisotropic' # 'RBF', 'IMQ'
    optimizer_type = 'LBFGS' # 'FullBatchLBFGS'

    svgd = SVGD(
        kernel_base_type=kernel_base_type,
        kernel_structure=None,
        median_heuristic=False,
        repulsive_scaling=2.,
        geom_metric_type='fisher',
    )

    particles, p_hist = svgd.apply(
                            particles,
                            model,
                            iters,
                            eps,
                            optimizer_type=optimizer_type,
                        )

    print("\nMean Est.: ", particles.mean(1))
    print("Std Est.: ", particles.std(1))

    create_movie_2D(
        p_hist,
        model.log_prob,
        to_numpy=True,
        save_path='./svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
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

    # ========= Matrix-valued SVGD ===================

    # # kernel_base_type = 'IMQ_Matrix'
    # kernel_base_type = 'RBF_Matrix'
    # # kernel_base_type = 'RBF_Weighted_Matrix'
    # # eps = 0.1
    # # eps = 5.
    # eps = 2.5
    # particles = particles_0.clone().cpu().numpy()
    # particles = torch.from_numpy(particles)
    #
    # matrix_svgd = MatrixSVGD(
    #     kernel_base_type=kernel_base_type,
    #     kernel_structure=None,
    #     # use_hessian_metric=True,
    #     use_hessian_metric=False,
    #     geom_metric_type='fisher',
    #     # geom_metric_type='full_hessian',
    #     # median_heuristic=True,
    #     )
    #
    # particles, p_hist = matrix_svgd.apply(
    #                         particles,
    #                         model,
    #                         iters,
    #                         eps,
    #                         optimizer_type='LBFGS'
    #                     )
    #
    # create_movie_2D(
    #     p_hist,
    #     model.log_prob,
    #     to_numpy=True,
    #     save_path='./matrix_RBF_svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #     # save_path='./weighted_matrix_RBF_svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #     # save_path='./check_matrix_RBF_svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #         kernel_base_type,
    #         num_particles,
    #         eps,
    #     ),
    #     opt='Matrix_SVGD',
    #     kernel_base_type=kernel_base_type,
    #     num_particles=num_particles,
    #     eps=eps,
    # )

    #============== Weighted Matrix-valued SVGD =====================
    # kernel_base_type = 'RBF_Matrix'
    #
    # particles = particles_0.clone().cpu().numpy()
    # particles = torch.from_numpy(particles)
    #
    # mix_matrix_svgd = MatrixMixtureSVGD(
    #     kernel_base_type=kernel_base_type,
    #     kernel_structure=None,
    #     geom_metric_type='full_hessian',
    #     # geom_metric_type='jacobian_product',
    #     )
    #
    # particles, p_hist = mix_matrix_svgd.apply(
    #                         particles,
    #                         model,
    #                         iters,
    #                         eps,
    #                         optimizer_type='LBFGS'
    #                     )
    #
    # create_movie_2D(
    #     p_hist,
    #     model.log_prob,
    #     to_numpy=True,
    #     save_path='./matrix_RBF_svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #     # save_path='./weighted_matrix_RBF_svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #     # save_path='./check_matrix_RBF_svgd_{}_gaussian_mix_np_{}_eps_{}.mp4'.format(
    #         kernel_base_type,
    #         num_particles,
    #         eps,
    #     ),
    #     opt='Matrix_Mix_SVGD',
    #     kernel_base_type=kernel_base_type,
    #     num_particles=num_particles,
    #     eps=eps,
    # )