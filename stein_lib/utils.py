import torch


def get_jacobian(
        gradient,
        X,
):
    """
    Returns the Jacobian matrix, given the gradient
    Parameters
    ----------
    gradient : (Tensor)
        Of shape [dim, batch]
    X : (Tensor)
        of shape [dim, batch]
    Returns
    -------
    J : (Tensor)
        Jacobian, of shape [dim, dim, batch]
    """
    dg_dXi = [
        torch.autograd.grad(
            gradient[:, i].sum(),
            X,
            retain_graph=True,
        )[0] for i in range(gradient.shape[1])
    ]
    J = torch.stack(dg_dXi, dim=1)
    return J


def calc_pw_distances(X):
    """
    Returns the pairwise distances between particles.
    Parameters
    ----------
    X : Tensor
        Points. of shape [dim, batch]
    """
    XX = X.matmul(X.t())
    pairwise_dists_sq = -2 * XX + XX.diag().unsqueeze(1) + XX.diag().unsqueeze(0)
    pw_dists = torch.sqrt(pairwise_dists_sq)
    return pw_dists