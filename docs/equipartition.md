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
any ASE dynamics object to track this. By default it reports the instantaneous
kinetic temperature of the whole system alongside each chemical species:

```py
from flashmd.ase.equipartition import EquipartitionMonitor

monitor = EquipartitionMonitor(atoms)
dyn.attach(monitor, interval=10)
dyn.run(1000)
```

You can also monitor extra custom groups (e.g. a spatial region) on top of the
per-species groups, and log every temperature to a file:

```py
monitor = EquipartitionMonitor(
    atoms,
    groups={"cluster": cluster_indices},
    logfile="equipartition.log",
)
dyn.attach(monitor, interval=10)
dyn.run(1000)
monitor.close()
```

Custom group names must not be ``"system"`` (reserved for the whole-system
temperature) or repeat an automatic per-species group name. Call
``monitor.close()`` once the run is done to close the log file.
