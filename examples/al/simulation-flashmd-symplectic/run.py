from typing import Callable
import torch
from metatomic.torch import load_atomistic_model
from ipi.utils.scripting import InteractiveSimulation
from flashmd.steppers import SymplecticStepper, FlashMDStepper
from flashmd.vv import flashmd_vv
from flashmd.wrappers import wrap_nvt
from flashmd.fpi import anderson_solver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("../input.xml", "r") as input_xml:
  sim = InteractiveSimulation(input_xml)

# load FlashMD model for initial guess
flashmd_model_32 = load_atomistic_model("../models/flashmd.pt")
flashmd_model_32.to(device)
initial_guess = FlashMDStepper(flashmd_model_32, device=device)

# load FlashMD symplectic model for corrector
flashmd_symplectic_model_32 = load_atomistic_model("../models/flashmd-symplectic.pt")
flashmd_symplectic_model_32.to(device)

# create a fixed-point solver and attach a logger to see the convergence behavior
solver_kwargs = dict(m=0, max_iter=100, tol=1e-3, beta=0.5)
def solver_with_log(
    g: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
) -> torch.Tensor:
    x_star, norms = anderson_solver(g, x0, return_residual_norms=True, **solver_kwargs)  # type: ignore
    print("l2 accuracies (converged in %d steps):" % len(norms))
    for i, n in enumerate(norms):
       print("iteration", i, "residual norm:", n)
    return x_star

# replace the motion step with a FlashMD stepper
stepper = SymplecticStepper(initial_guess, flashmd_symplectic_model_32, solver_with_log)
step_fn = flashmd_vv(sim, stepper, device=device, dtype=torch.float32, rescale_energy=False, random_rotation=False)
step_fn = wrap_nvt(sim, step_fn)
sim.set_motion_step(step_fn)

sim.run(100)
