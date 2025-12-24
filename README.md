FlashMD: universal long-stride molecular dynamics
=================================================

This repository contains custom integrators to run MD trajectories with FlashMD models. These models are
designed to learn and predict molecular dynamics trajectories using long strides, therefore allowing
very large time steps. Before using this method, make sure you are aware of its limitations, which are
discussed in [this preprint](http://arxiv.org/abs/2505.19350).

The pre-trained models we make available are trained to reproduce ab-initio MD at the r2SCAN level of theory.

Quickstart
----------

You can install the package with

```bash
  pip install flashmd
```

After installation, you can run accelerated molecular dynamics with ASE as follows:

```py
import ase.build
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import torch

from flashmd import get_pretrained
from flashmd.ase.langevin import Langevin


# Choose your time step (go for 10-30x what you would use in normal MD for your system)
time_step = 64  # 64 fs; also available: 1, 2, 4, 8, 16, 32, 128 fs

# Create a structure and initialize velocities
atoms = ase.build.bulk("Al", "fcc", cubic=True)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
atoms.set_velocities(  # it is generally a good idea to remove any net velocity
    atoms.get_velocities() - atoms.get_momenta().sum(axis=0) / atoms.get_masses().sum()
)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
energy_model, flashmd_model = get_pretrained("pet-omatpes", time_step)  

# Run MD
dyn = Langevin(
    atoms=atoms,
    timestep=time_step*ase.units.fs,
    temperature_K=300,
    time_constant=100*ase.units.fs,
    model=flashmd_model,
    device=device
)
dyn.run(1000)
```

[The first time you use this code and call the `get_pretrained` function, the
pre-trained models will be downloaded. This might take a bit.]

Other available integrators:

```py
  from flashmd.ase.velocity_verlet import VelocityVerlet
  from flashmd.ase.bussi import Bussi
```

Common pitfalls
---------------

Stick to 10-30x what you would use in normal MD for your system! The 64 fs example
above is good for metals. However,
- for most materials: try 32 fs (aggressive) or 16 fs (conservative)
- for aqueous and/or organic systems: try 16 fs (aggressive) or 8 fs (conservative)


Companion energy models
-----------------------

You might have noticed that ``get_pretrained()`` does not only return a FlashMD model,
but also an energy model. This is the energy model that the FlashMD model was trained
on. You might want to use it if...

Case 1: you want to run FlashMD with exact energy conservation, available through
``rescale_energy=True`` (this is enabled by default only when targeting the NVE
ensemble with ``VelocityVerlet``). In that case, before running FlashMD, you should
attach the energy calculator to the atoms:

```
from metatomic.torch.ase_calculator import MetatomicCalculator

...  # setting up atoms
calculator = MetatomicCalculator(energy_model, device=device, do_gradients_with_energy=False)
atoms.calc = calculator
...  # running FlashMD
```

Case 2: you want to compute energies after running FlashMD for your own analysis. In
this case, you can create the calculator just like in case 1 after running FlashMD.

Case 3: you found something interesting during a FlashMD run and you want to confirm it
with traditional MD. Then, you can just use ASE MD's modules as usual after attaching the
energy calculator:

```
from metatomic.torch.ase_calculator import MetatomicCalculator

...  # setting up atoms
calculator = MetatomicCalculator(energy_model, device=device)
atoms.calc = calculator
...  # running MD
```

In general, the energy models are slower and have a larger memory footprint compared to
the FlashMD models. As summarized above, you should use `do_gradients_with_energy=False`
to save computation and memory when you don't need forces.

Disclaimer
----------

This is experimental software and should only be used if you know what you're doing.
We recommend using the i-PI integrators for any serious work and/or if you need to perform
constant-pressure (NPT) molecular dynamics. You can see
[this cookbook recipe](https://atomistic-cookbook.org/examples/flashmd/flashmd-demo.html) 
for a usage example.
Given that the main issue we observe in direct MD trajectories is loss of equipartition
of energy between different degrees of freedom, we recommend using a local Langevin
thermostat, and to monitor the temperature of different atomic types or different
parts of the simulated system. 


Publication
-----------

If you found FlashMD useful, you can cite the corresponding article:

```
@article{FlashMD,
  title={FlashMD: long-stride, universal prediction of molecular dynamics},
  author={Bigi, Filippo and Chong, Sanggyu and Kristiadi, Agustinus and Ceriotti, Michele},
  journal={arXiv preprint arXiv:2505.19350},
  year={2025}
}
```

Reproducing the results in the article is supported with FlashMD v0.1.2:

```bash
pip install flashmd==0.1.2 ase==3.24.0 pet-mad==1.4.3
```

and using the "PET-MAD" models (PBEsol) from https://huggingface.co/lab-cosmo/flashmd.
Note that the results were obtained through the i-PI interface.

Instructions and material to reproduce the results in the paper are available on
Materials Cloud at https://doi.org/10.24435/materialscloud:b7-xq.
