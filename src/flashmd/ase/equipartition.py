import sys

import ase
import ase.units
import numpy as np
from ase.md.md import MolecularDynamics


class EquipartitionMonitor:
    """Track the kinetic temperature of each chemical species during an MD run.

    FlashMD's exact energy conservation (``rescale_energy=True``, see
    :class:`~flashmd.ase.velocity_verlet.VelocityVerlet`) is enforced by rescaling
    *all* atomic momenta by a single global factor each step. This fixes the total
    energy, but says nothing about how kinetic energy is distributed across degrees
    of freedom: this monitor reports the instantaneous kinetic temperature of the
    whole system alongside each chemical species. Attach it to an ASE dynamics
    object like any other observer::

        from flashmd.ase.equipartition import EquipartitionMonitor

        monitor = EquipartitionMonitor(dyn)
        dyn.attach(monitor, interval=10)

    Note:
        The ``"system"`` entry and the per-species entries can use different
        degrees-of-freedom conventions. ``"system"`` follows
        ``ase.Atoms.get_temperature()``, which counts ``3N`` degrees of freedom
        minus any removed by ``atoms.constraints`` (e.g. ``FixCom``), whereas
        every per-species entry always uses the full ``3 * n_species`` degrees
        of freedom regardless of such constraints, since a constraint on the
        whole system's center of mass can't be meaningfully attributed to an
        arbitrary subgroup. If ``atoms`` has no constraints, both conventions
        agree. If you do impose one (e.g. to remove center-of-mass drift), a
        small, systematic difference between ``"system"`` and the per-species
        values will appear from this convention mismatch alone, not from a real
        equipartition violation.

    Args:
        dyn: the dynamics object propagating the atoms, e.g. a
            :class:`~flashmd.ase.velocity_verlet.VelocityVerlet` instance. The
            monitor reads ``dyn.atoms`` for the current momenta and ``dyn.nsteps``
            to label the ``logfile`` step column correctly.
        logfile: if given, the system and per-species temperatures are written to
            this file (or to stdout if ``"-"``) every call, overwriting any
            existing content.
    """

    def __init__(
        self,
        dyn: MolecularDynamics,
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
