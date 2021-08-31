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
import matplotlib.pyplot as plt
import os.path as osp


class RBF_map:

    def __init__(self, xlim=(0,1), ylim=(0,1), delta=1, sigma=1, device=None):
        """
        :param xlim: x-limits of map
        :param ylim: y-limits of map
        :param delta: spacing between kernels
        :param sigma: standard deviation for each kernel
        """
        self.xlim=xlim; self.ylim=ylim;
        self.device = device
        self.feature_list = self.make_features(xlim, ylim, delta, sigma)
        # self.weights = torch.ones(len(self.feature_list), 1).double().to(device)
        self.weights = torch.ones(len(self.feature_list), 1).to(device)

    def make_features(self, xlim, ylim, delta, sigma):
        mu_list = self.make_xy_centers(xlim, ylim, delta)
        features = [self.kernel(mu, sigma) for mu in mu_list]

        # # DEBUG - single centered kernel ##
        # mu = torch.Tensor((0, 0)).to(self.device)
        # features = [self.kernel(mu, sigma)]

        features.append(self.base())
        return features

    def kernel(self, mu=torch.zeros(2), sigma=1):
        assert mu.dim() == 1 and mu.size(0) == 2
        return lambda x: torch.exp(-torch.bmm((x - mu).view(-1, 1, 2), (x - mu).view(-1, 2, 1)) / sigma**2).view(-1)

    def base(self):
        return lambda x: torch.ones(x.size(0), dtype=x.dtype, device=self.device).view(-1)

    def make_xy_centers(self, xlim, ylim, delta, offset=1e-3):
        """
        Make equally-spaced x-y means
        :return: list of tensor[(mu_x, mu_y)]
        """
        xc, yc = torch.meshgrid([torch.arange(xlim[0], xlim[1]+offset, delta), torch.arange(ylim[0], ylim[1]+offset, delta)])
        mu_vec = torch.stack((xc, yc), dim=2).view(-1,2).to(self.device)
        mu_list = [mu_vec[i,:] for i in range(mu_vec.shape[0])]
        return mu_list

    def eval(self, xy_data):
        return self.get_embedding(xy_data) @ self.weights

    def get_embedding(self, xy_data):
        return torch.stack([f(xy_data) for f in self.feature_list], dim=1)

    def fit(self, xy_grid, occ_grid):
        """
        Find weight parameters via linear regression.
        """
        xy_data = xy_grid.view(-1,2)
        labels = occ_grid.view(-1)
        Phi = self.get_embedding(xy_data)
        self.weights = torch.inverse(Phi.T.matmul(Phi)) @ (Phi.T @ labels)

    def fit_to_obst_map(self, obst_map, plot=False):
        """
        Implements RBF interpolation on an occupancy grid.
        """
        assert obst_map.xlim == self.xlim
        assert obst_map.ylim == self.ylim
        assert obst_map.map.size >= len(self.feature_list), "Number of map cells must be greater than number of RBF features."

        occ_grid = obst_map.convert_map()
        xy_grid = obst_map.get_xy_grid(device=self.device)
        if plot:
            obst_map.plot()
            self.plot()
        self.fit(xy_grid, occ_grid)
        if plot:
            self.plot()

    def get_collisions(self, X, device, clamp=False):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weights: weights on obstacle cost, float tensor
        :param X: Tensor of trajectories, of shape (batch_size, traj_length, position_dim)
        :return: collision cost on the trajectories
        """

        assert X.dim() == 3 and X.size(2) == 2
        batch_size, traj_length, _ = X.shape

        # Convert traj. positions to occupancy values
        # occ_values = self.eval(X.view(-1,2)).view(batch_size, traj_length, 1)
        occ_values = self.eval(X.reshape(-1,2)).view(batch_size, traj_length)
        if clamp: occ_values = torch.clamp(occ_values, 0, 1)
        return occ_values

    def make_costmap(self, xres=100, yres=100):
        # make grid points
        xlim = self.xlim; ylim = self.ylim
        xv, yv = torch.meshgrid([torch.linspace(xlim[0],xlim[1],xres), torch.linspace(ylim[0], ylim[1], yres)])
        # xy_grid = torch.stack((xv, yv), dim=2).double()
        xy_grid = torch.stack((xv, yv), dim=2)
        grid_shape = xy_grid.shape[:2]
        xy_vec = xy_grid.view(-1,2)
        # get cost
        out = self.eval(xy_vec.to(self.device))
        out = out.view(grid_shape).cpu().numpy()
        return out

    def plot(self, xres=100, yres=100, save_dir=None, filename="rbf_map.png"):
        # plot costmap
        out = self.make_costmap(xres, yres)
        plt.figure()
        plt.imshow(out)
        plt.gca().invert_yaxis()
        plt.show()
        if save_dir is not None:
            plt.savefig(osp.join(save_dir, filename))

        # plot cross-section
        # x_size, y_size = out.shape
        # plt.figure()
        # plt.plot(out[:,int(y_size*0.5)])
        # plt.plot(out[int(x_size*0.5),:])
        # plt.show()
