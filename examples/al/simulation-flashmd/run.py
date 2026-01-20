from metatomic.torch import load_atomistic_model
from ipi.utils.scripting import InteractiveSimulation
from flashmd.ipi import get_nvt_stepper

with open("../input.xml", "r") as input_xml:
  sim = InteractiveSimulation(input_xml)

# replace the motion step with a FlashMD stepper
flashmd_model_32 = load_atomistic_model("../models/flashmd.pt")
flashmd_model_32.to("cuda")
step_fn = get_nvt_stepper(sim, flashmd_model_32, "cuda")
sim.set_motion_step(step_fn)

sim.run(100)
