import torch
from metatomic.torch import load_atomistic_model
from ipi.utils.scripting import InteractiveSimulation
from flashmd.steppers import SymplecticStepper
from flashmd.stepper import FlashMDStepper
from flashmd.vv import flashmd_vv
from flashmd.wrappers import wrap_nvt

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

# replace the motion step with a FlashMD stepper
stepper = SymplecticStepper(initial_guess, flashmd_symplectic_model_32, None)
step_fn = flashmd_vv(sim, stepper, device=device, dtype=torch.float32, rescale_energy=False, random_rotation=False)
step_fn = wrap_nvt(sim, step_fn)
sim.set_motion_step(step_fn)

sim.run(100)
