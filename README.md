FlashMD: universal long-stride molecular dynamics
=================================================

This repository contains custom integrators to run MD trajectories with FlashMD models. These models are
designed to learn and predict molecular dynamics trajectories using long strides, therefore allowing
very large time steps. Before using this method, make sure you are aware of its limitations, which are
discussed in [this preprint](http://arxiv.org/abs/2505.19350).

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
from pet_mad.calculator import PETMADCalculator

from flashmd import get_universal_model
from flashmd.ase.langevin import Langevin


# Create a structure and initialize velocities
atoms = ase.build.bulk("Al", "fcc", cubic=True)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
calculator = PETMADCalculator("1.0.1", device=device)
atoms.calc = calculator
model = get_universal_model(16)  # 16 fs model; also available: 1, 4, 8, 32, 64 fs
model = model.to(device)

# Run MD
dyn = Langevin(
    atoms=atoms,
    timestep=16*ase.units.fs,
    temperature_K=300,
    time_constant=100*ase.units.fs,
    model=model,
    device=device
)
dyn.run(1000)
```

Other available integrators:

```py
  from flashmd.ase.velocity_verlet import VelocityVerlet
  from flashmd.ase.bussi import Bussi
```

Disclaimer
----------

This is experimental software and should only be used if you know what you're doing.
We recommend using the i-PI integrators for any serious work, and to perform constant
pressure, NpT molecular dynamics. You can see
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

and using the "PET-MAD" models from https://huggingface.co/lab-cosmo/flashmd. Note that
the results were obtained through the i-PI interface (as opposed to the ASE one).
