import ase.data
import ase.units
import numpy as np
import torch
from ipi.utils.depend import dstrip
from ipi.utils.mathtools import random_rotation as random_rotation_matrix
from ipi.utils.messages import info, verbosity

from .ipi import ipi_to_system, system_to_ipi
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
