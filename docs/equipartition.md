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

Note that the ``"system"`` and per-species entries can use different
degrees-of-freedom conventions. ``"system"`` follows
``ase.Atoms.get_temperature()``, which counts ``3N`` degrees of freedom minus
any removed by ``atoms.constraints`` (e.g. ``FixCom``), while every per-species
entry always uses the full ``3 * n_species`` degrees of freedom regardless of
such constraints, since a constraint on the whole system's center of mass can't
be meaningfully attributed to an arbitrary subgroup. With no constraints on
``atoms``, both conventions agree. If you impose one — for example to remove
center-of-mass drift — a small, systematic difference between ``"system"`` and
the per-species values will appear from this convention mismatch.
