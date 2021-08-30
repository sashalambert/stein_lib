import torch
import numpy as np


def get_graph(
        nodes,
        pw_dists,
        model,
        collision_res=0.1,
        prob_thresh=1.e-3,
        connect_radius=np.inf,
):
    # Get unique list of edges from upper triangle
    pw_dists = torch.triu(pw_dists)

    # Remove low-probability nodes (i.e. collision check)
    probs = torch.exp(model.log_prob(nodes))
    coll_inds = (probs <= prob_thresh).nonzero()
    pw_dists[coll_inds, :] = 0.
    pw_dists[:, coll_inds] = 0.

    # Filter edges according to max length
    pw_dists[pw_dists > connect_radius] = 0.

    # Get graph node indicies
    node_inds = (pw_dists > 0.).nonzero()

    

    # Collision check on edges:
    edges = node_inds
    graph = (nodes, edges)
    return graph