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
import pickle

def get_graph(
        particles,
        pw_dists,
        model,
        collision_res=0.1,
        collision_thresh=1.e-3,
        connect_radius=np.inf,
        include_coll_pts=False,
        save_path=None,
):
    # Avoid repeated edges
    pw_dists = torch.triu(pw_dists)

    # Remove low-probability nodes (i.e. collision check)
    # coll_vals = model.log_prob(particles)
    coll_vals = torch.exp(1 - model.log_prob(particles))
    coll_inds = (coll_vals > collision_thresh).nonzero()
    pw_dists[coll_inds, :] = 0.
    pw_dists[:, coll_inds] = 0.

    # Filter edges according to max length
    pw_dists[pw_dists > connect_radius] = 0.

    # Get graph node indicies
    node_inds = (pw_dists > 0.).nonzero()

    # Collision check on edges:
    edge_coll_pts = [] # for debugging
    edge_coll_vals = []
    edge_num_pts = []
    dim = particles.shape[-1]
    for node_pair in node_inds:
        start_pt = particles[node_pair[0]]
        end_pt = particles[node_pair[1]]
        edge_len = pw_dists[node_pair[0], node_pair[1]]
        num_pts = torch.floor(edge_len / collision_res).int()

        coll_pts = []
        for i in range(dim):
            coll_pts.append(torch.linspace(start_pt[i], end_pt[i], num_pts))  # includes endpoints

        coll_pts = torch.stack(coll_pts, dim=1)
        edge_cost = torch.exp(1 - model.log_prob(coll_pts)).sum()
        edge_coll_vals.append(edge_cost)
        edge_num_pts.append(num_pts)
        if include_coll_pts:
            edge_coll_pts.append(coll_pts) # for debugging

    edge_coll_vals = torch.stack(edge_coll_vals, dim=0)
    edge_coll_num_pts = torch.stack(edge_num_pts)

    if include_coll_pts:
        edge_coll_pts = torch.cat(edge_coll_pts, dim=0) # for debugging

    edge_lengths = pw_dists[node_inds]

    if not include_coll_pts:
        edge_coll_pts = None

    params = {
        'collision_res': collision_res,
        'collision_thresh': collision_thresh,
        'connect_radius': connect_radius,
    }

    graph = [
        node_inds,
        edge_lengths,
        edge_coll_vals,
        edge_coll_num_pts,
        edge_coll_pts,
        params,
    ]

    for i in range(len(graph)):
        try:
            graph[i] = graph[i].cpu().numpy()
        except:
            pass

    if save_path is not None:
        pickle.dump(graph, open(save_path, 'w'))

    return tuple(graph)