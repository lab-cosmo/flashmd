import ase.build
import ase.units
import pytest
import torch
from ase.md.velocitydistribution import thermalize_momenta

from flashmd import get_pretrained
from flashmd.ase.velocity_verlet import VelocityVerlet


@pytest.fixture(scope="module")
def models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, flashmd_model, symplectic_model = get_pretrained(
        "pet-omatpes", time_step=2, symplectic=True
    )
    return flashmd_model, symplectic_model, device


@pytest.fixture
def atoms():
    atoms = ase.build.bulk("Al", "fcc", cubic=True)
    thermalize_momenta(atoms, temperature_K=300)
    return atoms


def test_symplectic_without_config(atoms, models):
    """(flashmd_model, symplectic_model) runs without config."""
    flashmd_model, symplectic_model, device = models
    dyn = VelocityVerlet(
        atoms=atoms,
        timestep=2 * ase.units.fs,
        model=(flashmd_model, symplectic_model),
        device=device,
        rescale_energy=False,  # no energy calculator attached to atoms
    )
    dyn.run(3)


def test_symplectic_timestep_mismatch(atoms, models):
    """Pairing models with mismatched timesteps raises an error."""
    flashmd_model, _, device = models  # 2 fs FlashMD model
    _, _, symplectic_model_16fs = get_pretrained(
        "pet-omatpes", time_step=16, symplectic=True
    )
    with pytest.raises(ValueError, match="timestep"):
        VelocityVerlet(
            atoms=atoms,
            timestep=2 * ase.units.fs,
            model=(flashmd_model, symplectic_model_16fs),
            device=device,
            rescale_energy=False,
        )


def test_symplectic_with_config(atoms, models):
    """(flashmd_model, (symplectic_model, config)) runs with explicit config."""
    flashmd_model, symplectic_model, device = models
    dyn = VelocityVerlet(
        atoms=atoms,
        timestep=2 * ase.units.fs,
        model=(flashmd_model, (symplectic_model, {"tol": 1e-4, "max_iter": 10})),
        device=device,
        rescale_energy=False,  # no energy calculator attached to atoms
    )
    dyn.run(3)
