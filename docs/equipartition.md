Checking for equipartition violations
--------------------------------------

FlashMD's exact energy conservation (``rescale_energy=True``, the default for
``VelocityVerlet`` when targeting NVE, see [this guide](energy.md)) corrects the
*total* energy at every step by rescaling all atomic momenta by a single global
factor. This is not the same as enforcing the equipartition theorem: if the model's
per-step error is systematically biased towards some subset of degrees of freedom,
the global rescaling will lead to some atoms being systematically hotter or colder
than others.

``flashmd.ase.equipartition.EquipartitionMonitor`` is a diagnostic you can attach to
any ASE dynamics object to track this. It reports the instantaneous kinetic
temperature of the whole system alongside each chemical species:

```py
from flashmd.ase.equipartition import EquipartitionMonitor

monitor = EquipartitionMonitor(dyn)
dyn.attach(monitor, interval=10)
dyn.run(1000)
```

You can also log every temperature to a file:

```py
monitor = EquipartitionMonitor(dyn, logfile="equipartition.log")
dyn.attach(monitor, interval=10)
dyn.run(1000)
monitor.close()
```

Call ``monitor.close()`` once the run is done to close the log file.

Note that the ``"system"`` entry and the per-species entries use different
degrees-of-freedom conventions. Many MD codes (e.g. LAMMPS, i-PI) subtract 3
degrees of freedom from the *global* temperature to account for the conserved
center-of-mass motion, but cannot meaningfully apply that correction to an
arbitrary subgroup, since a subgroup's own center of mass isn't separately
conserved. Here, ``"system"`` follows ``ase.Atoms.get_temperature()`` (``3N``
degrees of freedom, unless you have added a constraint that removes some),
while every per-species entry always uses the full ``3 * n_species`` degrees of
freedom. A small, systematic difference between ``"system"`` and the
per-species values (or against an external code's own diagnostic) can come from
this convention mismatch alone, not necessarily a real equipartition violation.
