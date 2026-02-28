import ase.build
import ase.units
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator
from flashmd.ase.velocity_verlet import VelocityVerlet


def test_isolated_atom(monkeypatch, tmp_path):
    """Test that a short MD run completes without errors on an isolated atom."""
    monkeypatch.chdir(tmp_path)

    atoms = ase.Atoms("O", positions=[[0, 0, 0]])
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    time_step = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_model, flashmd_model = get_pretrained("pet-omatpes-v2", time_step)
    calculator = EnergyCalculator(energy_model, device=device)
    atoms.calc = calculator

    dyn = VelocityVerlet(
        atoms=atoms,
        timestep=time_step * ase.units.fs,
        model=flashmd_model,
        device=device,
    )
    dyn.run(10)


def test_slab_plus_isolated_atom(monkeypatch, tmp_path):
    """Test that a short MD run completes without errors on a slab plus an isolated atom."""
    monkeypatch.chdir(tmp_path)

    # Create a slab and an isolated atom
    slab = ase.build.fcc111("Al", size=(2, 2, 3), vacuum=10)
    isolated_atom = ase.Atoms("O", positions=[[0, 0, 24]])
    atoms = slab + isolated_atom
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    time_step = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_model, flashmd_model = get_pretrained("pet-omatpes-v2", time_step)
    calculator = EnergyCalculator(energy_model, device=device)
    atoms.calc = calculator

    dyn = VelocityVerlet(
        atoms=atoms,
        timestep=time_step * ase.units.fs,
        model=flashmd_model,
        device=device,
    )
    dyn.run(10)
