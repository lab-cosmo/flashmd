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
