from typing import Callable, List

import torch


@torch.jit.script
def _anderson_update(
    x: torch.Tensor,
    g: torch.Tensor,
    delta_xs: List[torch.Tensor],
    delta_gs: List[torch.Tensor],
    beta: float,
    lambda_reg: float,
) -> torch.Tensor:
    """One Anderson-accelerated fixed-point step."""
    if len(delta_xs) > 0:
        X = torch.stack(delta_xs, dim=1)
        G = torch.stack(delta_gs, dim=1)
        A = G.T @ G + lambda_reg * torch.eye(G.shape[1], device=G.device, dtype=g.dtype)
        coeffs = torch.linalg.solve(A, G.T @ g)
        return x + beta * g - (X + beta * G) @ coeffs
    else:
        return x + beta * g


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
    delta_xs: list[torch.Tensor] = []
    delta_gs: list[torch.Tensor] = []
    residual_norms = []

    x = x0
    fx = f(x)
    g = fx - x
    x_prev, g_prev = None, None
    for k in range(max_iter):
        res_norm = torch.norm(g).item()
        residual_norms.append(res_norm)
        if res_norm < tol:
            break

        if k > 0:
            assert x_prev is not None and g_prev is not None
            delta_xs.append(x - x_prev)
            delta_gs.append(g - g_prev)
            if len(delta_xs) > m:
                delta_xs.pop(0)
                delta_gs.pop(0)
        x_prev, g_prev = x, g

        x = _anderson_update(x, g, delta_xs, delta_gs, beta, lambda_reg)

        fx = f(x)
        g = fx - x

    if return_residual_norms:
        return x, residual_norms
    else:
        return x
