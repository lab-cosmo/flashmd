FlashMD: universal long-stride molecular dynamics
=================================================

This repository contains custom integrators to run MD trajectories with FlashMD models. These models are
designed to learn and predict molecular dynamics trajectories using long strides, therefore allowing
very large time steps. When using this method, make sure you are aware of its limitations, which are
discussed in [this preprint](http://arxiv.org/abs/2505.19350).

The pre-trained models we make available are trained to reproduce ab-initio MD at the r2SCAN level of theory.

ASE Quickstart (see below for LAMMPS)
-------------------------------------

You can install the package with

```bash
  pip install flashmd
```

After installation, you can run accelerated molecular dynamics as follows:

```py
import ase.build
import ase.units
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from flashmd import get_pretrained
from flashmd.ase import EnergyCalculator
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
device = "cuda" if torch.cuda.is_available() else "cpu"
energy_model, flashmd_model = get_pretrained("pet-omatpes-v2", time_step)

# Set the energy model (optional, see below for more precise usage)
calculator = EnergyCalculator(energy_model, device=device)
atoms.calc = calculator

# Run MD
dyn = Langevin(
    atoms=atoms,
    timestep=time_step*ase.units.fs,
    temperature_K=300,
    time_constant=100*ase.units.fs,
    model=flashmd_model,
    device=device,
)
dyn.run(1000)  # this is 64 ps!
```

[The first time you use this code and call the `get_pretrained` function, the
pre-trained models will be downloaded. This might take a bit.]

Other available integrators:

```py
  from flashmd.ase.velocity_verlet import VelocityVerlet
  from flashmd.ase.bussi import Bussi
```

Along with all FlashMD models, we also provide the potential energy model whose
dynamics they are trained to reproduce. See this short [guide](docs/energy.md) on how to
best use the energy models if you want to enforce exact energy conservation during
FlashMD runs, run traditional MD with the energy model, and more.

Common pitfalls
---------------

Stick to 10-30x what you would use in normal MD for your system! The 64 fs example
above is good for metals. However,
- for most materials: try 32 fs (aggressive) or 16 fs (conservative)
- for aqueous and/or organic systems: try 16 fs (aggressive) or 8 fs (conservative)

Using FlashMD in LAMMPS
-----------------------

LAMMPS can allow you to run FlashMD with better computational and memory efficiency.
Furthermore, it will give you access to more sophisticated types of simulations, such as
simulations in the NPT ensemble and metadynamics. See [here](docs/lammps.md) for a guide
on using FlashMD in LAMMPS.

Using FlashMD in i-PI
---------------------

You can see
[this cookbook recipe](https://atomistic-cookbook.org/examples/flashmd/flashmd-demo.html) 
for usage examples. i-PI is our most mature interface, and the one that was used to
generate all our published results.

Models
------

See [here](docs/models.md) for the complete list of the models we provide. If you are
new to FlashMD, we recommend starting with the ``pet-omatpes-v2`` models.

Training/fine-tuning your own FlashMD models
--------------------------------------------

FlashMD models can be trained from the **metatrain library**. This
[tutorial](https://docs.metatensor.org/metatrain/latest/generated_examples/1-advanced/04-flashmd.html)
shows how to train your own FlashMD model, either from scratch or by fine-tuning of one of our universal models.

The trajectory datasets generated with the PET-MAD baseline MLIPs used in the paper can be found in [this
repository](https://zenodo.org/records/17904449).

Disclaimer
----------

This is experimental software and should only be used if you know what you are doing.
Given that the main issue we observe in FlashMD is loss of equipartition
of energy between different degrees of freedom, we recommend using a Langevin
thermostat, possibly monitoring the temperature of different atomic types or different
parts of the simulated system. For the time being, we also recommend checking all
FlashMD-powered findings with traditional MD. The energy models that were used to train
FlashMD, and that we make available in this repository, can be used for this purpose.

#### More energy models

Do you want to try out different energy models (MLIPs),
trained on different levels of theory and/or with different accuracy-speed tradeoffs? Check
out the **[UPET](https://github.com/lab-cosmo/upet) repository.**

Publication
-----------

If you found FlashMD useful, you can cite the corresponding [article](https://arxiv.org/abs/2505.19350):

```
@inproceedings{FlashMD,
  title     = {FlashMD: long-stride, universal prediction of molecular dynamics},
  author    = {Bigi, Filippo and Chong, Sanggyu and Kristiadi, Agustinus and Ceriotti, Michele},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.19350}
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
