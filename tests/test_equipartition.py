import ase
import ase.build
import ase.units
import numpy as np
import pytest
import torch
from ase.md.velocitydistribution import thermalize_momenta

from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator
from flashmd.ase.equipartition import EquipartitionMonitor
from flashmd.ase.velocity_verlet import VelocityVerlet


class _FakeDyn:
    """Minimal stand-in for an ASE dynamics object, exposing just the two
    attributes EquipartitionMonitor relies on."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.nsteps = 0


def _equilibrated_atoms(temperature_K=300.0, seed=0):
    """Build a heteroatomic system with all species at the same temperature.

    Uses a large enough supercell so instantaneous per-species kinetic
    temperatures are not dominated by statistical noise.
    """
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (3, 3, 3)
    rng = np.random.default_rng(seed)
    masses = atoms.get_masses()
    momenta = rng.standard_normal((len(atoms), 3)) * np.sqrt(
        masses[:, None] * ase.units.kB * temperature_K
    )
    atoms.set_momenta(momenta)
    return atoms


def test_reports_one_group_per_species_by_default():
    dyn = _FakeDyn(_equilibrated_atoms())
    monitor = EquipartitionMonitor(dyn)

    assert set(monitor.groups) == {"species:Na", "species:Cl"}
    report = monitor()
    assert set(report) == {"system", "species:Na", "species:Cl"}
    assert monitor.history == [report]


def test_custom_groups_are_added_alongside_species_groups():
    atoms = _equilibrated_atoms()
    dyn = _FakeDyn(atoms)
    monitor = EquipartitionMonitor(dyn, groups={"first_half": range(len(atoms) // 2)})

    assert set(monitor.groups) == {"species:Na", "species:Cl", "first_half"}
    report = monitor()
    assert set(report) == {"system", "species:Na", "species:Cl", "first_half"}


def test_group_name_system_is_rejected():
    dyn = _FakeDyn(_equilibrated_atoms())
    with pytest.raises(ValueError, match="system"):
        EquipartitionMonitor(dyn, groups={"system": [0]})


def test_group_name_clashing_with_species_group_is_rejected():
    dyn = _FakeDyn(_equilibrated_atoms())
    with pytest.raises(ValueError, match="species:Na"):
        EquipartitionMonitor(dyn, groups={"species:Na": [0]})


def test_logfile_records_every_call(tmp_path):
    dyn = _FakeDyn(_equilibrated_atoms())
    logfile = tmp_path / "equipartition.log"
    monitor = EquipartitionMonitor(dyn, logfile=str(logfile))

    monitor()
    monitor()
    monitor.close()

    lines = logfile.read_text().splitlines()
    assert lines[0].startswith("#")
    assert len(lines) == 3  # header + two recorded steps


def test_logfile_step_column_matches_dynamics_step(tmp_path):
    """The logged step must reflect the real dynamics step count, not just how
    many times the monitor has been called (which would be wrong whenever the
    monitor is attached with interval != 1)."""
    dyn = _FakeDyn(_equilibrated_atoms())
    logfile = tmp_path / "equipartition.log"
    monitor = EquipartitionMonitor(dyn, logfile=str(logfile))

    monitor()
    dyn.nsteps = 10
    monitor()
    dyn.nsteps = 20
    monitor()
    monitor.close()

    lines = logfile.read_text().splitlines()
    steps = [int(line.split()[0]) for line in lines[1:]]
    assert steps == [0, 10, 20]


def test_logfile_overwrites_stale_content(tmp_path):
    logfile = tmp_path / "equipartition.log"
    logfile.write_text("stale content from a previous run\n")

    dyn = _FakeDyn(_equilibrated_atoms())
    monitor = EquipartitionMonitor(dyn, logfile=str(logfile))
    monitor()
    monitor.close()

    lines = logfile.read_text().splitlines()
    assert "stale content from a previous run" not in lines
    assert lines[0].startswith("#")


def test_attaches_as_a_dynamics_observer(monkeypatch, tmp_path):
    """The monitor should work as a plain ASE dynamics observer, no model required
    for the monitor itself; a short FlashMD run exercises the actual attach point.
    """
    monkeypatch.chdir(tmp_path)

    atoms = ase.build.bulk("Al", "fcc", cubic=True)
    thermalize_momenta(atoms, temperature_K=300)

    time_step = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_model, flashmd_model = get_pretrained("pet-omatpes-v2", time_step)
    atoms.calc = EnergyCalculator(energy_model, device=device)

    dyn = VelocityVerlet(
        atoms=atoms,
        timestep=time_step * ase.units.fs,
        model=flashmd_model,
        device=device,
    )
    monitor = EquipartitionMonitor(dyn, groups={"all": range(len(atoms))})
    dyn.attach(monitor, interval=1)
    dyn.run(5)

    # ASE calls observers once for the initial state plus once per step.
    assert len(monitor.history) == 6
