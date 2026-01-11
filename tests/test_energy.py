import ase.build
import ase.io
import ase.units
import torch
from ase.md import VelocityVerlet

from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator


def test_md(monkeypatch, tmp_path):
    """Test that a short MD run completes without errors with a Trajectory file."""
    monkeypatch.chdir(tmp_path)

    atoms = ase.build.bulk("Al", "fcc", cubic=True)

    time_step = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_model, _ = get_pretrained("pet-omatpes-v2", time_step)
    calculator = EnergyCalculator(
        energy_model, device=device, do_gradients_with_energy=False
    )
    atoms.calc = calculator

    dyn = VelocityVerlet(atoms=atoms, timestep=time_step * ase.units.fs)
    traj = ase.io.Trajectory("test_md.traj", "w", atoms)
    dyn.attach(traj.write)
    dyn.run(10)
    traj.close()
