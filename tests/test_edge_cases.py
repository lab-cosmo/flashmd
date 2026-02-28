import ase.build
import ase.io
import ase.units
import torch
from ase.md import VelocityVerlet

from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator


def test_isolated_atom(monkeypatch, tmp_path):
    """Test that a short MD run completes without errors on an isolated atom."""
    monkeypatch.chdir(tmp_path)

    atoms = ase.Atoms("O", positions=[[0, 0, 0]])

    time_step = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_model, _ = get_pretrained("pet-omatpes-v2", time_step)
    calculator = EnergyCalculator(energy_model, device=device)
    atoms.calc = calculator

    dyn = VelocityVerlet(atoms=atoms, timestep=time_step * ase.units.fs)
    dyn.run(10)


def test_slab_plus_isolated_atom(monkeypatch, tmp_path):
    """Test that a short MD run completes without errors on a slab plus an isolated atom."""
    monkeypatch.chdir(tmp_path)

    # Create a slab and an isolated atom
    slab = ase.build.fcc111("Al", size=(2, 2, 3), vacuum=10)
    isolated_atom = ase.Atoms("O", positions=[[0, 0, 24]])
    atoms = slab + isolated_atom

    time_step = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_model, _ = get_pretrained("pet-omatpes-v2", time_step)
    calculator = EnergyCalculator(energy_model, device=device)
    atoms.calc = calculator

    dyn = VelocityVerlet(atoms=atoms, timestep=time_step * ase.units.fs)
    dyn.run(10)
