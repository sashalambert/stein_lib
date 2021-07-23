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
from svgd.matrix_svgd.matrix_svgd import MatrixSVGD

from stein_lib.models import doubleBanana_analytic
from stein_lib.utils import create_movie_2D

torch.set_default_tensor_type(torch.DoubleTensor)

# Params
num_particles = 50
# num_particles = 3
iters = 100
# eps_list = [0.5]
# eps_list = [0.1]
# eps_list = [0.01, 0.1]
# kernel_base_type = 'RBF'
kernel_base_type = 'RBF_Anisotropic'

# Sample intial particles
torch.manual_seed(0)
# prior_dist = Normal(loc=0., scale=1.5)
# prior_dist = Normal(loc=0., scale=0.75)
prior_dist = Normal(loc=0., scale=0.5)
particles_0 = prior_dist.sample((2, num_particles))

# DEBUG - load particles
# particles_0 = np.load('/home/sasha/np_50.npy')
# particles_0 = torch.from_numpy(particles_0).t()

# Load model
model = doubleBanana_analytic(
    mu_n=3.57857342, # actually, your observation
    # mu_n=np.log(30), # actually, your observation
    seed=0,
)


 ### SVN  ########
#
# particles = particles_0.clone().cpu().numpy()
# particles = torch.from_numpy(particles)
#
# eps = 0.1
# kernel_base_type='RBF_Anisotropic'
# # kernel_base_type='RBF'
# # kernel_base_type='IMQ'
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
#                         use_analytic_grads=True,
#                     )
#
# create_movie_2D(
#     p_hist,
#     model.log_prob,
#     ax_limits=(-2, 2),
#     to_numpy=True,
#     save_path='./svn_tests/svn_{}_double_banana_x_np_{}_eps_{}.mp4'.format(
#         kernel_base_type,
#         num_particles,
#         eps,
#     ),
#     opt='SVN',
#     kernel_base_type=kernel_base_type,
#     num_particles=num_particles,
#     eps=eps,
# )

#### SVGD ############

# particles = particles_0.clone().cpu().numpy()
# particles = torch.from_numpy(particles)
# eps = 1.

# svgd = SVGD(
#     kernel_base_type=kernel_base_type,
#     kernel_structure=None,
#     geom_metric_type='full_hessian',
#     )
#
# particles, p_hist = svgd.apply(
#                         particles,
#                         model,
#                         iters,
#                         eps,
#                         use_analytic_grads=True,
#                     )
#
# create_movie_2D(
#     p_hist,
#     model.log_prob,
#     ax_limits=(-2, 2),
#     to_numpy=True,
#     save_path='./svgd_{}_double_banana_np_{}_eps_{}.mp4'.format(
#         kernel_base_type,
#         num_particles,
#         eps,
#     ),
#     opt='SVGD',
#     kernel_base_type=kernel_base_type,
#     num_particles=num_particles,
#     eps=eps,
# )

# #### Matrix-valued SVGD ############

# kernel_base_type = 'IMQ_Matrix'
kernel_base_type = 'RBF_Matrix'
# eps = 1.
# eps = 2.5
eps = 10.
particles = particles_0.clone().cpu().numpy()
particles = torch.from_numpy(particles)

matrix_svgd = MatrixSVGD(
    kernel_base_type=kernel_base_type,
    kernel_structure=None,
    use_hessian_metric=True,
    # use_hessian_metric=False,
    # geom_metric_type='fisher',
    geom_metric_type='full_hessian',
    # median_heuristic=True,
    )

particles, p_hist = matrix_svgd.apply(
                        particles,
                        model,
                        iters,
                        eps,
                        use_analytic_grads=False,
                    )

create_movie_2D(
    p_hist,
    model.log_prob,
    ax_limits=(-2, 2),
    to_numpy=True,
    save_path='./matrix_svgd_{}_double_banana_np_{}_eps_{}.mp4'.format(
        kernel_base_type,
        num_particles,
        eps,
    ),
    opt='Matrix_SVGD',
    kernel_base_type=kernel_base_type,
    num_particles=num_particles,
    eps=eps,
)

#### Weighted Matrix-valued SVGD ############

# kernel_base_type = 'RBF_Matrix'
# # eps = 1.
# eps = 0.5
# # eps = 0.1
#
# particles = particles_0.clone().cpu().numpy()
# particles = torch.from_numpy(particles)
#
# mix_matrix_svgd = MatrixMixtureSVGD(
#     kernel_base_type=kernel_base_type,
#     kernel_structure=None,
#     geom_metric_type='full_hessian',
#     hessian_scale=1.,
#     )
#
# particles, p_hist = mix_matrix_svgd.apply(
#                         particles,
#                         model,
#                         iters,
#                         eps,
#                         use_analytic_grads=True,
#                     )
#
# create_movie_2D(
#     p_hist,
#     model.log_prob,
#     ax_limits=(-2, 2),
#     to_numpy=True,
#     save_path='./mix_matrix_RBF_svgd_{}_double_banana_np_{}_eps_{}.mp4'.format(
#         kernel_base_type,
#         num_particles,
#         eps,
#     ),
#     opt='Matrix_Mix_SVGD',
#     kernel_base_type=kernel_base_type,
#     num_particles=num_particles,
#     eps=eps,
# )