from ipi.utils.scripting import InteractiveSimulation
from flashmd import get_pretrained
from flashmd.ipi import get_nvt_stepper

with open("../input.xml", "r") as input_xml:
  sim = InteractiveSimulation(input_xml)

# replace the motion step with a FlashMD stepper
_, flashmd_model_32 = get_pretrained("pet-omatpes", 32)
step_fn = get_nvt_stepper(sim, flashmd_model_32, "cuda")
sim.set_motion_step(step_fn)

sim.run(100)
