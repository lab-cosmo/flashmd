import torch

from flashmd.fpi import anderson_solver


def test_anderson_solver_convergence():
    """Test that the Anderson solver converges on a simple fixed-point problem."""

    def f(x):
        return 0.5 * x + 1.0

    x0 = torch.tensor([0.0])
    x_sol, residuals = anderson_solver(
        f, x0, m=3, max_iter=100, tol=1e-6, return_residual_norms=True
    )
    x_exact = torch.tensor([2.0])

    assert torch.allclose(x_sol, x_exact, atol=1e-5)
    assert all(earlier >= later for earlier, later in zip(residuals, residuals[1:]))
