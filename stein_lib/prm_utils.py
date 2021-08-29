import torch


def get_graph(X, pw_dists, thresh=None, coll_res=None):

    nodes = X

    # Get unique list of edges from upper triangle
    pw_dists = torch.triu(pw_dists)

    node_inds = (pw_dists > 0.).nonzero()

    edges = []


    graph = (nodes, edges)
    return graph