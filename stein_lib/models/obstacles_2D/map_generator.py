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

from obs_map import ObstacleRectangle, ObstacleMap
from rbf_map import RBF_map
from obs_utils import random_rect, save_map_image


def generate_obstacle_map(
        map_dim=(10,10),
        obst_list=[],
        cell_size=1.,
        start_pts=None,
        goal_pts=None,
        seed=0,
        device=None,
        random_gen=False,
        num_obst=0,
        rand_xy_limits=None,
        rand_shape=[2,2],
        map_type=None,
        plot=False,
        delta=0.5,
        sigma=0.5,
):

    """
    Args
    ---
    map_dim : (int,int)
        2D tuple containing dimensions of obstacle/occupancy grid.
        Treat as [x,y] coordinates. Origin is in the center.
        ** Dimensions must be an even number. **
    cell_sz : float
        size of each square map cell
    obst_list : [(cx_i, cy_i, width, height)]
        List of obstacle param tuples
    start_pts : float
        Array of x-y points for start configuration.
        Dim: [Num. of points, 2]
    goal_pts : float
        Array of x-y points for target configuration.
        Dim: [Num. of points, 2]
    seed : int or None
    random_gen : bool
        Specify whether to generate random obstacles. Will first generate obstacles provided by obst_list,
        then add random obstacles until number specified by num_obst.
    num_obst : int
        Total number of obstacles
    rand_limit: [[float, float],[float, float]]
        List defining x-y sampling bounds [[x_min, x_max], [y_min, y_max]]
    rand_shape: [float, float]
        Shape [width, height] of randomly generated obstacles.
    """
    ## Make occpuancy grid
    obst_map = ObstacleMap(map_dim, cell_size, device=device)
    num_fixed = len(obst_list)
    for param in obst_list:
        cx, cy, width, height = param
        rect = ObstacleRectangle(cx,cy,width,height)
        ## Check validity of new obstacle
        # valid = rect._obstacle_collision_check(obst_map)
        # rect._point_collision_check(obst_map,start_pts) & \
        # rect._point_collision_check(obst_map,goal_pts)
        rect._add_to_map(obst_map)

    ## Add random obstacles
    if random_gen:
        # random.seed(seed)
        assert num_fixed <= num_obst, "Total number of obstacles must be greater than or equal to number specified in obst_list"
        xlim = rand_xy_limits[0]
        ylim = rand_xy_limits[1]
        width = rand_shape[0]
        height = rand_shape[1]
        for _ in range(num_obst - num_fixed + 1):
            num_attempts = 0
            max_attempts = 25
            while num_attempts <= max_attempts:
                rect = random_rect(xlim, ylim, width, height)

                # Check validity of new obstacle
                valid = rect._obstacle_collision_check(obst_map)
                # rect._point_collision_check(obst_map,start_pts) & \
                # rect._point_collision_check(obst_map,goal_pts)

                if valid:
                    # Add to Map
                    rect._add_to_map(obst_map)
                    # Add to list
                    obst_list.append([rect.center_x,rect.center_y,
                                      rect.width, rect.height])
                    break

                if num_attempts == max_attempts:
                    print("Obstacle generation: Max. number of attempts reached. ")
                    print("Total num. obstacles: {}.  Num. random obstacles: {}.\n"
                          .format( len(obst_list), len(obst_list) - num_fixed))

                num_attempts += 1

    obst_map.convert_map()

    ## Fit mapping model
    if map_type == 'RBF':
        print("Generating RBF map...\n")
        rbf_map = RBF_map(obst_map.xlim, obst_map.ylim, delta=delta, sigma=sigma, device=device)
        rbf_map.fit_to_obst_map(obst_map, plot=plot)
        return rbf_map

    elif map_type == 'direct':
        return obst_map

    else:
        raise IOError('Map type "{}" not recognized'.format(map_type))
    return obst_list


if __name__ == "__main__":

    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    obst_list = [(0, 0, 4, 8)]
    cell_size = 0.1
    map_dim = [20, 20]
    seed = 0

    obst_map = generate_obstacle_map(
        map_dim, obst_list, cell_size,
        map_type='RBF',
        random_gen=True,
        # random_gen=False,
        num_obst=5,
        rand_xy_limits=[[-10, 10], [-10, 10]],
        rand_shape=[2,2],
        plot=True,
    )

    traj_y = torch.linspace(-map_dim[1]/2., map_dim[1]/2., 20)
    traj_x = torch.zeros_like(traj_y)
    X = torch.cat((traj_x.unsqueeze(1), traj_y.unsqueeze(1)), dim=1)
    cost = obst_map.eval(X)
    print(cost)

    # Get cost:

    # obst_map = generate_obstacle_map(
    #     map_dim, obst_list, cell_size,
    #     map_type='direct',
    #     seed=seed,
    #     random_gen=True,
    #     num_obst=5,
    #     rand_xy_limits=[[-10, 10], [-10, 10]],
    #     rand_shape=[2,2],
    # )
    # obst_map.plot()
    # save_map_image(obst_map.map,dir='/tmp')

    # w_obs = 1.
    # traj_y = torch.linspace(-map_dim[1]/2., map_dim[1]/2., 20)
    # traj_x = torch.zeros_like(traj_y)
    # X = torch.stack([traj_x, traj_y], dim=1)
    # X = X.repeat(2,1,1)
    # obst_cost, occ_values = obst_map.get_collision_cost(w_obs, X.double(), device=torch.device('cpu'))
    # print(X)
