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
            gradient[i].sum(),
            X,
            retain_graph=True,
        )[0] for i in range(gradient.shape[0])
    ]
    J = torch.stack(dg_dXi, dim=0)
    return J
