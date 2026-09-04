Companion energy models and exact energy conservation
-----------------------------------------------------

You might have noticed that ``get_pretrained()`` does not only return a FlashMD model,
but also an energy model, which is itself just a machine-learned interatomic potential.
This is the energy model that the FlashMD model was trained on. You might want to use it
if...

**Case 1**: you want to run FlashMD with exact energy conservation, available through the
parameter ``rescale_energy=True`` in the FlashMD integrator (this is enabled by
default only when targeting the NVE ensemble with ``VelocityVerlet``). In that case,
besides setting this flag, you should attach the energy calculator to the atoms before
running FlashMD, exactly as shown in the opening example (and below with the more precise
``do_gradients_with_energy=False`` which will save you memory and computation):

```
from flashmd.ase import EnergyCalculator

...  # setting up atoms
calculator = EnergyCalculator(energy_model, device=device, do_gradients_with_energy=False)
atoms.calc = calculator
...  # running FlashMD
```

**Case 2**: you want to compute energies after running FlashMD for your own analysis. In
this case, you can create the calculator just like in case 1, but possibly after running
FlashMD and/or in a different script.

**Case 3**: you found something interesting during a FlashMD run and you want to confirm it
with traditional MD. Then, you can just use ASE's MD modules as usual after attaching
the energy calculator:

```
from flashmd.ase import EnergyCalculator

...  # setting up atoms
calculator = EnergyCalculator(energy_model, device=device)
atoms.calc = calculator
...  # running MD
```

In general, the energy models are slower and have a larger memory footprint compared to
the FlashMD models. As summarized above, you should use `do_gradients_with_energy=False`
to save computation and memory when you do not need forces.

Monitoring the rescaling factor
--------------------------------

Every time ``rescale_energy=True`` triggers a rescale, the momenta are multiplied by a
factor ``alpha = sqrt(1 - (E_new - E_old) / E_kin)``. If a step increases the total energy
by more than the post-step kinetic energy can absorb, no real ``alpha`` exists to restore
energy conservation; rather than silently producing ``NaN`` momenta, both the ASE and i-PI
integrators raise a ``RuntimeError`` in that case. This is a sign that the step was
unphysical (e.g. atomic overlap or a model extrapolation error) -- consider using a
smaller time step.

You can also monitor ``alpha`` directly, to catch large corrections before they become
outright failures.

**ASE**: the last computed value is available as ``dyn.alpha`` (``None`` until the first
rescaled step). Attach an observer to log it during a run:

```
dyn.attach(lambda: print(dyn.alpha), interval=1)
```

**i-PI**: the value is written each step to ``motion.flashmd_alpha`` (``nan`` on any step
where rescaling did not run), which you can expose as a genuine column in the ``.out``
file, next to volume, pressure, etc., by registering it as a custom property when you
build the ``InteractiveSimulation``:

```
sim = InteractiveSimulation(
    input_xml,
    custom_properties={
        "flashmd_alpha": {
            "func": lambda self: getattr(self.motion, "flashmd_alpha", float("nan")),
            "dimension": "undefined",
            "help": "FlashMD momentum rescale factor (energy conservation).",
        }
    },
)
```

and adding ``flashmd_alpha`` to the ``<properties>`` list in your xml file. The custom
property must be registered this way, in the Python script that builds the
``InteractiveSimulation``, before you can reference it in the xml: i-PI validates the
``<properties>`` list against its property registry as soon as the simulation object is
built, so adding ``flashmd_alpha`` to the xml alone, without this registration, raises
``KeyError: flashmd_alpha is not a recognized property``.
