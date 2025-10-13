import ase.build
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import torch
from pet_mad.calculator import PETMADCalculator

from metatomic.torch import load_atomistic_model
from flashmd.ase.langevin import Langevin


# Create a structure and initialize velocities
atoms = ase.build.molecule("H2O")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
calculator = PETMADCalculator("1.0.1", device=device)
atoms.calc = calculator

model = load_atomistic_model("flashmd.pt")
model = model.to(device)

# Run MD
dyn = Langevin(
    atoms=atoms,
    timestep=5.0*ase.units.fs,
    temperature_K=300,
    time_constant=100*ase.units.fs,
    model=model,
    device=device,
    rescale_energy=False,
)
dyn.run(1000)
