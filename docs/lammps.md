Running FlashMD in LAMMPS
=========================

Installation
------------

Running in LAMMPS is supported starting from FlashMD 0.2.5. Make sure you have an
updated version:

```bash
    pip install flashmd --upgrade
```

Next, you will need to install a version of LAMMPS with FlashMD enabled. Here, we focus
on installing it quickly using conda with GPU acceleration via Kokkos. For other
installation configurations (compiling from source, CPU-only, etc.), as well as more
details, you can refer to this
[page](https://docs.metatensor.org/metatomic/latest/engines/lammps.html#how-to-install-the-code).

**Step 1** Find your GPU's architecture, either via the command
``nvidia-smi --query-gpu=compute_cap --format=csv,noheader`` or by looking it up
from this [page](https://developer.nvidia.com/cuda/gpus). Remove the . (dot) from the
number you get and match it to the number from the following list: 
``VOLTA70, AMPERE80, AMPERE86, ADA89, HOPPER90``. For example, if you got 8.9, go for
"ADA89".

**Step 2** Run ``conda install -c metatensor -c conda-forge "lammps-metatomic=*=cuda*ADA89*nompi*"``,
where you should replace ``ADA89`` with the string you found above. If you do not have
conda yet, we recommend the [Miniforge](https://github.com/conda-forge/miniforge) conda provider.
On HPC systems, it is safer to execute this command on a GPU-enabled compute (or debug)
node, as sometimes Nvidia drivers are not present on login nodes and this can prevent
conda from installing the correct GPU libraries.

You are now ready to run FlashMD in LAMMPS!

**Note:** FlashMD in LAMMPS does not support running on multiple GPUs yet, and we are
actively working on this feature. However, FlashMD supports between 120k and 300k atoms
even on a single GPU on modern HPC clusters.

**Note:** FlashMD in LAMMPS does not support running with exact energy conservation. You
can use the i-PI and ASE interfaces if you need this functionality.

Usage
-----

LAMMPS will need the FlashMD model and, in some cases, the energy model (MLIP) it was
trained on. Here is how you can get them in the current directory from Python:

```py
    from flashmd import get_pretrained

    time_step = 16  # in fs, also available: 1, 2, 4, 8, 32, 64, 128
    energy_model, flashmd_model = get_pretrained("pet-omatpes-v2", time_step)
    energy_model.save("mlip.pt")
    flashmd_model.save(f"flashmd-{time_step}.pt")
```

**Note:** Stick to no more than 30x the time step that you would use in normal MD for your system!

Here below you will see how to run different types of molecular dynamics. In all cases,
you should launch LAMMPS as ``lmp -in in.flashmd -k on g 1 -pk kokkos newton on neigh half -sf kk``
(assuming your input file is named ``in.flashmd``), or ``lmp -in in.flashmd`` if you
want to run without Kokkos acceleration. The following sections will present some input
files that you can take inspiration from. 

# NVT (Langevin thermostat)

Langevin dynamics is conceptually one of the easiest methods to target the NVT ensemble
in molecular dynamics. Here is an input file to use a Langevin thermostat with FlashMD:

```
# Create a block of Al -- replace it with your system
units           metal
atom_style      atomic
boundary        p p p
lattice         fcc 4.05
region          box block 0 5 0 5 0 5
create_box      1 box
create_atoms    1 box

# Setting the energy model is useful if you want to have a look at the energy, but it
# will slow down your simulation. Uncomment it if you need it.
# pair_style      metatomic mlip.pt device cuda non_conservative on
# pair_coeff      * * 13

mass            1 26.9815386

timestep        0.016  # 16 fs here, must match the time step of the FlashMD model!

velocity        all create 700.0 12345 mom yes rot yes dist gaussian

fix             0 all metatomic flashmd-16.pt types 13 device cuda
fix             1 all langevin 700.0 700.0 0.1 12345

thermo          10

run             1000  # runs 16 ps
```

Here, the "13" appearing in ``pair_coeff`` and ``fix metatomic`` is the atomic number of
Al. If you have a system with multiple LAMMPS types, you will need to replace "13" by the
atomic numbers corresponding to the LAMMPS atom types, in order (e.g., often "8 1" for
water simulations).

The same goes for the mass. For example, for water,
```
mass 1 15.9994
mass 2 1.008
```
FlashMD is not transferable across different masses. Therefore, if your atomic masses do
not correspond to those used to train FlashMD (which are the standard atomic weights,
i.e., abundance-weighted isotopic averages), an error will be raised. The same will
happen if your time step does not match that of the FlashMD model. 

# NVT (CSVR thermostat)

Although coupling a system with a Langevin thermostat samples the NVT ensemble
correctly, it can alter the dynamical properties of the simulation. In contrast, the
CSVR thermostat results in NVT sampling while leaving the dynamical properties largely
unaffected.

```
# Create a block of Al -- replace it with your system
units           metal
atom_style      atomic
boundary        p p p
lattice         fcc 4.05
region          box block 0 5 0 5 0 5
create_box      1 box
create_atoms    1 box

# Setting the energy model is useful if you want to have a look at the energy, but it
# will slow down your simulation. Uncomment it if you need it.
# pair_style      metatomic mlip.pt device cuda non_conservative on
# pair_coeff      * * 13

mass            1 26.9815386

timestep        0.016  # 16 fs here, must match the time step of the FlashMD model!

velocity        all create 700.0 12345 mom yes rot yes dist gaussian

fix             0 all metatomic flashmd-16.pt types 13 device cuda
fix             1 all temp/csvr 700.0 700.0 0.1 12345

thermo          10

run             1000  # runs 16 ps
```

Beware! When using the CSVR thermostat, although FlashMD will still very often be
qualitatively correct, the dynamical properties of the simulation might be
quantitatively off, especially when using large time steps. When using this thermostat,
you should confirm any findings with normal MD: uncomment the ``pair_style`` and
``pair_coeff`` lines, remove ``non_conservative on``, replace the ``fix metatomic`` line
with ``fix 0 all nve`` and you will be ready to run traditional MD with the r2SCAN model
that FlashMD was trained on. 

# NPT

To run NPT, just add a pressure-control fix to the simulation. For example:

```
# Create a block of Al -- replace it with your system
units           metal
atom_style      atomic
boundary        p p p
lattice         fcc 4.05
region          box block 0 5 0 5 0 5
create_box      1 box
create_atoms    1 box

# Setting the energy model is useful if you want to have a look at the energy, but it
# will slow down your simulation. Uncomment it if you need it.
pair_style      metatomic mlip.pt device cuda
pair_coeff      * * 13

mass            1 26.9815386

timestep        0.016  # 16 fs here, must match the time step of the FlashMD model!

velocity        all create 700.0 12345 mom yes rot yes dist gaussian

fix             0 all metatomic flashmd-16.pt types 13 device cuda
fix             1 all langevin 700.0 700.0 0.1 12345  # or CSVR
fix             2 all press/langevin iso 1.0 1.0 1.0 temp 700.0 700.0 67890

thermo          10

run             1000  # runs 16 ps
```

Note that this time we uncommented the lines defining the energy model:
```
pair_style      metatomic mlip.pt device cuda
pair_coeff      * * 13
```
(and we removed ``non_conservative on``). This is because the energy model is used to
provide stresses for pressure control. This will slow your simulation down quite a
bit, but we are actively working to make NPT dynamics with FlashMD more efficient!

# Metadynamics using PLUMED

To run metadynamics with PLUMED, you can just add a PLUMED fix to the simulation.
For example:

```
fix             0 all metatomic flashmd-16.pt types 13 device cuda
fix             1 all langevin 700.0 700.0 0.1 12345
fix             2 all plumed plumedfile plumed.dat outfile plumed.log
```
