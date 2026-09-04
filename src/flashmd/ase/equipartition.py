import sys
from collections.abc import Sequence

import ase
import ase.units
import numpy as np
from ase.md.md import MolecularDynamics


class EquipartitionMonitor:
    """Track the kinetic temperature of groups of atoms during an MD run.

    FlashMD's exact energy conservation (``rescale_energy=True``, see
    :class:`~flashmd.ase.velocity_verlet.VelocityVerlet`) is enforced by rescaling
    *all* atomic momenta by a single global factor each step. This fixes the total
    energy, but says nothing about how kinetic energy is distributed across degrees
    of freedom: this monitor reports the instantaneous kinetic temperature of the
    whole system and of one group per chemical species, plus any extra custom
    groups you provide. Attach it to an ASE dynamics object like any other observer::

        from flashmd.ase.equipartition import EquipartitionMonitor

        monitor = EquipartitionMonitor(dyn, groups={"cluster": [0, 1, 5]})
        dyn.attach(monitor, interval=10)

    Args:
        dyn: the dynamics object propagating the atoms, e.g. a
            :class:`~flashmd.ase.velocity_verlet.VelocityVerlet` instance. The
            monitor reads ``dyn.atoms`` for the current momenta and ``dyn.nsteps``
            to label the ``logfile`` step column correctly.
        groups: extra named groups of atom indices to report the temperature of,
            on top of the automatic per-species groups (e.g. ``{"cluster": [0, 1,
            5]}`` for a spatial region). Groups may overlap and need not cover all
            atoms. A group name must not be ``"system"`` or clash with an
            automatic per-species group name.
        logfile: if given, the system and group temperatures are written to this
            file (or to stdout if ``"-"``) every call, overwriting any existing
            content.
    """

    def __init__(
        self,
        dyn: MolecularDynamics,
        groups: dict[str, Sequence[int]] | None = None,
        logfile: str | None = None,
    ):
        self.dyn = dyn
        self.atoms = dyn.atoms

        symbols = self.atoms.get_chemical_symbols()
        self.groups: dict[str, np.ndarray] = {
            f"species:{symbol}": np.array(
                [i for i, s in enumerate(symbols) if s == symbol], dtype=int
            )
            for symbol in sorted(set(symbols))
        }
        for name, indices in (groups or {}).items():
            if name == "system" or name in self.groups:
                raise ValueError(
                    f"group name {name!r} is reserved or already in use "
                    "(names must not be 'system' and must not repeat an "
                    "automatic per-species group name)"
                )
            self.groups[name] = np.asarray(indices, dtype=int)

        self._logfile = None
        if logfile == "-":
            self._logfile = sys.stdout
        elif logfile is not None:
            self._logfile = open(logfile, "w")
        self._header_written = False

        self.history: list[dict[str, float]] = []

    def _group_temperature(self, indices: np.ndarray) -> float:
        momenta = self.atoms.get_momenta()[indices]
        masses = self.atoms.get_masses()[indices]
        kinetic_energy = 0.5 * np.sum(momenta**2 / masses[:, None])
        n_degrees_of_freedom = 3 * len(indices)
        return 2.0 * kinetic_energy / (n_degrees_of_freedom * ase.units.kB)

    def __call__(self) -> dict[str, float]:
        report = {"system": self.atoms.get_temperature()}
        report.update(
            (name, self._group_temperature(indices))
            for name, indices in self.groups.items()
            if len(indices) > 0
        )
        self.history.append(report)

        if self._logfile is not None:
            if not self._header_written:
                self._logfile.write("# step  " + "  ".join(report) + "\n")
                self._header_written = True
            values = "  ".join(f"{report[name]:.2f}" for name in report)
            self._logfile.write(f"{self.dyn.nsteps}  {values}\n")
            self._logfile.flush()

        return report

    def close(self) -> None:
        """Close the log file, if one was opened."""
        if self._logfile is not None and self._logfile is not sys.stdout:
            self._logfile.close()
