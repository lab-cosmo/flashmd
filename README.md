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
from metatomic.torch.ase_calculator import MetatomicCalculator

from flashmd import get_pretrained
from flashmd.ase.langevin import Langevin


# Choose your time step (go for 10-30x what you would use in normal MD for your system)
time_step = 16  # 16 fs; also available: 1, 2, 4, 8, 32, 64, 128 fs

# Create a structure and initialize velocities
atoms = ase.build.bulk("Al", "fcc", cubic=True)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
energy_model, flashmd_model = get_pretrained("pet-omatpes", time_step)  

calculator = MetatomicCalculator(energy_model, device=device)
atoms.calc = calculator

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
