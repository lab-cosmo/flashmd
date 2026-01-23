import torch
from ipi.utils.scripting import InteractiveSimulation
from flashmd import get_pretrained
from flashmd.steppers import FlashMDStepper
from flashmd.wrappers import wrap_nvt
from flashmd.vv import flashmd_vv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("../input.xml", "r") as input_xml:
  sim = InteractiveSimulation(input_xml)

# replace the motion step with a FlashMD stepper
_, flashmd_model_32 = get_pretrained("pet-omatpes", 32)
stepper = FlashMDStepper(flashmd_model_32, device=device)
step_fn = flashmd_vv(sim, stepper, device=device, dtype=torch.float32, rescale_energy=False)
step_fn = wrap_nvt(sim, step_fn)
sim.set_motion_step(step_fn)

sim.run(100)
