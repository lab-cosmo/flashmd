Checking for equipartition violations
--------------------------------------

FlashMD's exact energy conservation (``rescale_energy=True``, the default for
``VelocityVerlet`` when targeting NVE, see [this guide](energy.md)) corrects the
*total* energy at every step by rescaling all atomic momenta by a single global
factor. This is not the same as enforcing the equipartition theorem: if the model's
per-step error is systematically biased towards some subset of degrees of freedom
(for example a minority species, or a spatial region), the global rescaling
preserves that bias instead of correcting it. Total energy will look perfectly
conserved while some atoms run systematically hotter or colder than others, which
can be mistaken for a real physical effect (e.g. a "hot" minority species
triggering spurious nucleation in a disordered system).

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
```

Custom group names must not be ``"system"`` (reserved for the whole-system
temperature) or repeat an automatic per-species group name.

Because the monitor reports *instantaneous* temperatures, small groups fluctuate a
lot from statistics alone (a group with only a handful of atoms can easily swing by
tens of percent even in a perfectly equilibrated system) — apply your own averaging
to ``monitor.history`` after the run before drawing conclusions.
