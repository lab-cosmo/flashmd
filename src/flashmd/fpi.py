from typing import Callable

import torch


def anderson_solver(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    m: int = 5,
    max_iter: int = 50,
    tol: float = 1e-5,
    beta: float = 1.0,
    lambda_reg: float = 1e-4,
    return_residual_norms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
    """
    Solve fixed-point problem x = f(x) using Anderson acceleration.

    Args:
        f: Fixed-point mapping.
        x0: Initial guess.
        m: Number of previous iterates to use for acceleration.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance based on residual norm.
        beta: Mixing parameter for the fixed-point step.
        lambda_reg: Regularization parameter for least-squares solve.
        return_residual_norms: If True, also return list of residual norms.

    Returns:
        Approximate solution x, and optionally list of residual norms.
    """
    # history buffers
    delta_xs, delta_gs = [], []
    residual_norms = []

    # run fixed-pointer iteration
    x = x0
    fx = f(x)
    g = fx - x
    x_prev, g_prev = None, None
    for k in range(max_iter):
        # evaluate residual and compute convergence
        res_norm = torch.norm(g).item()
        residual_norms.append(res_norm)
        if res_norm < tol:
            break

        # update history
        if k > 0:
            assert x_prev is not None and g_prev is not None
            delta_xs.append(x - x_prev)
            delta_gs.append(g - g_prev)

            # truncate history to hold at most m elements
            if len(delta_xs) > m:
                delta_xs.pop(0)
                delta_gs.pop(0)
        x_prev, g_prev = x, g

        # compute Anderson acceleration step
        if len(delta_xs) > 0:
            # create matrices from history of shape (features, history_length)
            X = torch.stack(delta_xs, dim=1)  # (n, k)
            G = torch.stack(delta_gs, dim=1)  # (n, k)

            # solve regularized least-squares problem
            A = G.T @ G + lambda_reg * torch.eye(G.shape[1], device=G.device)
            b = G.T @ g
            try:
                coeffs = torch.linalg.solve(A, b)
                # update iterate with momentum + Anderson step
                x = x + beta * g - (X + beta * G) @ coeffs
            except RuntimeError:
                x = x + beta * g  # fallback to fixed-point step if matrix is singular
        else:
            x = x + beta * g  # fixed-point step if there is no history

        # update iterate and residual
        fx = f(x)
        g = fx - x

    if return_residual_norms:
        return x, residual_norms
    else:
        return x
