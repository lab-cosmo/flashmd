"""
This is a self-contained example to show how to use different integrators.

First, we run a simple MD simulation for Al and log the trajectory. Then, we process
the trajectory into two datasets: one for training a FlashMD model and one for training
a symplectic FlashMD model. Then, we train both models and run dynamics with them with
i-PI.
"""

# %%
import shutil
import subprocess
import torch
from ase import Atoms
import ase.io
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from upet.calculator import UPETCalculator
from tqdm import trange
from metatomic.torch import load_atomistic_model
from ipi.utils.scripting import InteractiveSimulation
from flashmd.steppers import FlashMDStepper, SymplecticStepper
from flashmd.fpi import anderson_solver
from flashmd.vv import flashmd_vv
from flashmd.wrappers.nve import wrap_nve

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
# Create a bulk Al system for demonstation.
atoms = bulk("Al", "fcc", cubic=True) * (3, 3, 3)
ase.io.write("al.xyz", atoms)
len(atoms)

# %%
# Attach a UPET calculator
atoms.calc = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cuda")
# TODO: export the model with mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.5.0.ckpt
atoms.get_potential_energy()

# %%
# Set up a simulation and equilibrate
MaxwellBoltzmannDistribution(atoms, temperature_K=400)
Stationary(atoms)
gamma = 1 / (200 * units.fs)
Langevin(atoms, 2 * units.fs, temperature_K=400, friction=gamma).run(1000)

# %%
# Run NVE MD with ASE for an Al system.
mlip_integrator = VelocityVerlet(atoms, 2 * units.fs)
structures = []
for _ in trange(1000):
  mlip_integrator.run(1)
  structures.append(atoms.copy())

# %%
# Write the trajectory to an easy-to-use format.
ase.io.write("al.xyz", structures)

# %%
# Preprocess the trajectories to be readable for both versions of FlashMD. This code is
# laregly equal to the code in metatrain showing how to train the various models.

structures: list[Atoms] = ase.io.read("al.xyz", index=":") # type: ignore
i = 0
num_step_frames = 4
num_decorrelation_frames = 10
assert num_decorrelation_frames > 1
flashmd_structures = []
symplectic_flashmd_structures = []
while i < len(structures) - num_step_frames + 1:
  # TODO: add reverse augmentation

  # Extract the current and future positions and momenta.
  current_q = structures[i].get_positions()
  current_p = structures[i].get_momenta()
  future_q = structures[i + num_step_frames - 1].get_positions()
  future_p = structures[i + num_step_frames - 1].get_momenta()

  # For FlashMD, take a frame and frame + num_step_frames ahead.
  flashmd_structure = structures[i].copy()
  flashmd_structure.arrays["future_positions"] = future_q
  flashmd_structure.arrays["future_momenta"] = future_p
  flashmd_structures.append(flashmd_structure)

  # For symplectic FlashMD, the input is a midpoint and the target is the delta between
  # the start and the end configuration.
  symplectic_flashmd_structure = structures[i].copy()
  symplectic_flashmd_structure.set_positions((current_q + future_q) / 2)
  symplectic_flashmd_structure.set_momenta((current_p + future_p) / 2)
  symplectic_flashmd_structure.arrays["delta_positions"] = future_q - current_q
  symplectic_flashmd_structure.arrays["delta_momenta"] = future_p - current_p
  symplectic_flashmd_structures.append(symplectic_flashmd_structure)

  i += num_decorrelation_frames
print(f"{len(flashmd_structures)=}, {len(symplectic_flashmd_structures)=}")

# %%
# Write the processed frames to two dataset files.
ase.io.write("start-to-end.xyz", flashmd_structures)
ase.io.write("midpoint-to-delta.xyz", symplectic_flashmd_structures)

# %%
# Train models with the datasets.
subprocess.run(["mtt", "train", "options-flashmd.yaml"], check=True)
shutil.move("model.pt", "flashmd.pt")
subprocess.run(["mtt", "train", "options-symplectic-flashmd.yaml"], check=True)
shutil.move("model.pt", "symplectic-flashmd.pt")

# %%
# Load the input file template for i-PI. We replace the motion step later with an
# appropriate FlashMD step function.
with open("simulation-template.xml") as f:
  input_template = f.read()

# %%
# Run NVE dynamics with i-PI and FlashMD
flashmd = load_atomistic_model("flashmd.pt").to(device)
stepper = FlashMDStepper(flashmd, device=device)
simulation = InteractiveSimulation(input_template.replace("PREFIX", "flashmd"))
step_fn = flashmd_vv(simulation, stepper, device=device, dtype=torch.float32)
step_fn = wrap_nve(simulation, step_fn)
simulation.set_motion_step(step_fn)
simulation.run(100)

# %%
# %% Run NVE dynamics with i-PI and symplectic FlashMD
symplectic_flashmd = load_atomistic_model("symplectic-flashmd.pt").to(device)
symplectic_stepper = SymplecticStepper(stepper, symplectic_flashmd, anderson_solver)
symplectic_simulation = InteractiveSimulation(input_template.replace("PREFIX", "symplectic-flashmd"))
symplectic_step_fn = flashmd_vv(symplectic_simulation, symplectic_stepper, device=device, dtype=torch.float32)
symplectic_step_fn = wrap_nve(symplectic_simulation, symplectic_step_fn)
symplectic_simulation.set_motion_step(symplectic_step_fn)
symplectic_simulation.run(100)
