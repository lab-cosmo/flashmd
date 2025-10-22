import ase.build
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import torch
from pet_mad.calculator import PETMADCalculator
from metatomic.torch.ase_calculator import MetatomicCalculator

from flashmd import get_pretrained
from flashmd.ase.langevin import Langevin


# Create a structure and initialize velocities
atoms = ase.build.bulk("Al", "fcc", cubic=True)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
energy_model, flashmd_model = get_pretrained("pet-omatpes", 16)  # 16 fs model; also available: 1, 4, 8, 32, 64 fs

calculator = MetatomicCalculator(energy_model, device=device)
atoms.calc = calculator

# Run MD
dyn = Langevin(
    atoms=atoms,
    timestep=16*ase.units.fs,
    temperature_K=300,
    time_constant=100*ase.units.fs,
    model=flashmd_model,
    device=device
)
dyn.run(1000)
