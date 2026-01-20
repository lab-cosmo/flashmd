import ase.data
import ase.units
import numpy as np
import torch
from ipi.utils.depend import dstrip
from ipi.utils.mathtools import random_rotation as random_rotation_matrix
from ipi.utils.messages import info, verbosity
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .steppers.flashmd import AtomisticStepper


def standard_vv(sim, rescale_energy: bool = False):
    """
    Returns a velocity Verlet stepper function for i-PI simulations.

    Parameters:
        sim: The i-PI simulation object.
        rescale_energy: If True, rescales the kinetic energy after the step
            to maintain energy conservation.

    Returns:
        A function that performs a velocity Verlet step.
    """

    def vv_step(motion):
        old_energy = None
        if rescale_energy:
            info("@flashmd: Old energy", verbosity.debug)
            old_energy = sim.properties("potential") + sim.properties("kinetic_md")

        print(motion.integrator.pdt, motion.integrator.qdt)
        motion.integrator.pstep(level=0)
        motion.integrator.pconstraints()
        motion.integrator.qcstep()  # does two steps because qdt is halved in the i-PI integrator
        motion.integrator.qcstep()
        motion.integrator.pstep(level=0)
        motion.integrator.pconstraints()

        if rescale_energy:
            info("@flashmd: Energy rescale", verbosity.debug)
            new_energy = sim.properties("potential") + sim.properties("kinetic_md")
            kinetic_energy = sim.properties("kinetic_md")
            alpha = np.sqrt(1.0 - (new_energy - old_energy) / kinetic_energy)
            motion.beads.p[:] = alpha * dstrip(motion.beads.p)

    return vv_step


def flashmd_vv(
    sim,
    stepper: AtomisticStepper,
    device: torch.device,
    dtype: torch.dtype,
    rescale_energy=True,
    random_rotation=False,
):
    # compare the model's internal timestep with the i-PI one -- they need to match
    dt = sim.syslist[0].motion.dt * 2.4188843e-17 * ase.units.s
    timestep = stepper.get_timestep()
    if not np.allclose(dt, timestep):
        raise ValueError(
            f"Mismatch between timestep ({dt}) and model timestep ({timestep})."
        )
    
    def flashmd_vv(motion):
        info("@flashmd: Starting VV", verbosity.debug)
        old_energy = None
        if rescale_energy:
            info("@flashmd: Old energy", verbosity.debug)
            old_energy = sim.properties("potential") + sim.properties("kinetic_md")

        info("@flashmd: Stepper", verbosity.debug)
        system = ipi_to_system(motion, device, dtype)

        R = None
        if random_rotation:
            # generate a random rotation matrix
            R = torch.tensor(
                random_rotation_matrix(motion.prng, improper=True),
                device=system.positions.device,
                dtype=system.positions.dtype,
            )
            # applies the random rotation
            system.cell = system.cell @ R.T
            system.positions = system.positions @ R.T
            momenta = system.get_data("momenta").block(0).values.squeeze()
            momenta[:] = momenta @ R.T  # does the change in place

        print(system)
        new_system = stepper.step(system)

        if random_rotation:
            # revert q,p to the original reference frame (`system_to_ipi` ignores the cell)
            new_system.positions = new_system.positions @ R
            momenta = new_system.get_data("momenta").block(0).values.squeeze()
            momenta[:] = momenta @ R

        info("@flashmd: System to ipi", verbosity.debug)
        system_to_ipi(motion, new_system)
        info("@flashmd: VV P constraints", verbosity.debug)
        motion.integrator.pconstraints()

        if rescale_energy:
            info("@flashmd: Energy rescale", verbosity.debug)
            new_energy = sim.properties("potential") + sim.properties("kinetic_md")
            kinetic_energy = sim.properties("kinetic_md")
            alpha = np.sqrt(1.0 - (new_energy - old_energy) / kinetic_energy)
            motion.beads.p[:] = alpha * dstrip(motion.beads.p)
        motion.integrator.pconstraints()
        info("@flashmd: End of VV step", verbosity.debug)

    return flashmd_vv


def ipi_to_system(motion, device, dtype):
    positions = (
        dstrip(motion.beads.q).reshape(-1, 3) * ase.units.Bohr / ase.units.Angstrom
    )
    positions_torch = torch.tensor(positions, device=device, dtype=dtype)
    cell = dstrip(motion.cell.h).T * ase.units.Bohr / ase.units.Angstrom
    cell_torch = torch.tensor(cell, device=device, dtype=dtype)
    pbc_torch = torch.tensor([True, True, True], device=device, dtype=torch.bool)
    momenta = (
        dstrip(motion.beads.p).reshape(-1, 3)
        * (9.1093819e-31 * ase.units.kg)
        * (ase.units.Bohr / ase.units.Angstrom)
        / (2.4188843e-17 * ase.units.s)
    )
    momenta_torch = torch.tensor(momenta, device=device, dtype=dtype)
    masses = dstrip(motion.beads.m) * 9.1093819e-31 * ase.units.kg
    masses_torch = torch.tensor(masses, device=device, dtype=dtype)
    types_torch = torch.tensor(
        [ase.data.atomic_numbers[name] for name in motion.beads.names],
        device=device,
        dtype=torch.int32,
    )
    system = System(types_torch, positions_torch, cell_torch, pbc_torch)
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=momenta_torch.unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(len(momenta_torch))], device=device
                        ),
                    ),
                    components=[
                        Labels(
                            names="xyz",
                            values=torch.tensor([[0], [1], [2]], device=device),
                        )
                    ],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )
    system.add_data(
        "masses",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=masses_torch.unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(len(masses_torch))], device=device
                        ),
                    ),
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )
    return system


def system_to_ipi(motion, system):
    # only needs to convert positions and momenta, it's assumed that the cell won't be changed
    motion.beads.q[:] = (
        system.positions.detach().cpu().numpy().flatten()
        * ase.units.Angstrom
        / ase.units.Bohr
    )
    motion.beads.p[:] = system.get_data("momenta").block().values.detach().squeeze(
        -1
    ).cpu().numpy().flatten() / (
        (9.1093819e-31 * ase.units.kg)
        * (ase.units.Bohr / ase.units.Angstrom)
        / (2.4188843e-17 * ase.units.s)
    )
