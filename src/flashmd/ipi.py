from ipi.utils.depend import dstrip
from ipi.utils.units import Constants
from ipi.utils.messages import verbosity, info
from ipi.utils.mathtools import random_rotation as random_rotation_matrix
from ipi.engine.motion.dynamics import NVEIntegrator, NVTIntegrator, NPTIntegrator

from flashmd.stepper import FlashMDStepper
import ase.units
import torch
import numpy as np
import ase.data

from metatomic.torch import System
from metatensor.torch import Labels, TensorBlock, TensorMap


def get_standard_vv_step(
    sim, model=None, device=None, rescale_energy=True, random_rotation=False
):
    """
    Returns a velocity Verlet stepper function for i-PI simulations.

    Parameters:
    - sim: The i-PI simulation object.
    - rescale_energy: If True, rescales the kinetic energy after the step
        to maintain energy conservation.

    Returns:
    - A function that performs a velocity Verlet step.
    """

    def vv_step(motion):
        if random_rotation:
            raise NotImplementedError(
                "Random rotation is not implemented in the standard VV stepper."
            )

        if rescale_energy:
            info("@flashmd: Old energy", verbosity.debug)
            old_energy = sim.properties("conserved")

        motion.integrator.pstep(level=0)
        motion.integrator.pconstraints()
        motion.integrator.qcstep()  # does two steps because qdt is halved in the i-PI integrator
        motion.integrator.qcstep()
        motion.integrator.pstep(level=0)
        motion.integrator.pconstraints()

        if rescale_energy:
            info("@flashmd: Energy rescale", verbosity.debug)
            new_energy = sim.properties("conserved")
            kinetic_energy = sim.properties("kinetic_md")
            alpha = np.sqrt(1.0 - (new_energy - old_energy) / kinetic_energy)
            motion.beads.p[:] = alpha * dstrip(motion.beads.p)

    return vv_step


def get_flashmd_vv_step(sim, model, device, rescale_energy=True, random_rotation=False):
    capabilities = model.capabilities()

    model_timestep = float(model.module.timestep)

    dt = sim.syslist[0].motion.dt * 2.4188843e-17 * ase.units.s / ase.units.fs

    if not np.allclose(dt, model_timestep):
        raise ValueError(
            f"Mismatch between timestep ({dt} fs) and model timestep ({model_timestep} fs)."
        )

    device = torch.device(device)
    dtype = getattr(torch, capabilities.dtype)
    stepper = FlashMDStepper(model, device)

    def flashmd_vv(motion):
        info("@flashmd: Starting VV", verbosity.debug)
        if rescale_energy:
            info("@flashmd: Old energy", verbosity.debug)
            old_energy = sim.properties("conserved")

        info("@flashmd: Stepper", verbosity.debug)
        system = ipi_to_system(motion, device, dtype)

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
            new_energy = sim.properties("conserved")
            kinetic_energy = sim.properties("kinetic_md")
            alpha = np.sqrt(1.0 - (new_energy - old_energy) / kinetic_energy)
            motion.beads.p[:] = alpha * dstrip(motion.beads.p)
            motion.integrator.pconstraints()  # just to make sure; should be cheap

        info("@flashmd: End of VV step", verbosity.debug)

    return flashmd_vv


def get_nve_stepper(
    sim,
    model,
    device,
    rescale_energy=True,
    random_rotation=False,
    use_standard_vv=False,
):
    motion = sim.syslist[0].motion
    if type(motion.integrator) is not NVEIntegrator:
        raise TypeError(
            f"Base i-PI integrator is of type {motion.integrator.__class__.__name__}, use a NVE setup."
        )

    if use_standard_vv:
        # use the standard velocity Verlet integrator
        vv_step = get_standard_vv_step(
            sim, model, device, rescale_energy, random_rotation
        )
    else:
        # defaults to the FlashMD VV stepper
        vv_step = get_flashmd_vv_step(
            sim, model, device, rescale_energy, random_rotation
        )

    def nve_stepper(motion, *_, **__):
        vv_step(motion)
        motion.ensemble.time += motion.dt

    return nve_stepper


def get_nvt_stepper(
    sim,
    model,
    device,
    rescale_energy=True,
    random_rotation=False,
    use_standard_vv=False,
):
    motion = sim.syslist[0].motion
    if type(motion.integrator) is not NVTIntegrator:
        raise TypeError(
            f"Base i-PI integrator is of type {motion.integrator.__class__.__name__}, use a NVT setup."
        )

    if use_standard_vv:
        # use the standard velocity Verlet integrator
        vv_step = get_standard_vv_step(
            sim, model, device, rescale_energy, random_rotation
        )
    else:
        # defaults to the FlashMD VV stepper
        vv_step = get_flashmd_vv_step(
            sim, model, device, rescale_energy, random_rotation
        )

    def nvt_stepper(motion, *_, **__):
        # OBABO splitting of a NVT propagator
        motion.thermostat.step()
        motion.integrator.pconstraints()
        vv_step(motion)
        motion.thermostat.step()
        motion.integrator.pconstraints()
        motion.ensemble.time += motion.dt

    return nvt_stepper


def _qbaro(baro):
    """Propagation step for the cell volume (adjusting atomic positions and momenta)."""

    v = baro.p[0] / baro.m[0]
    halfdt = (
        baro.qdt
    )  # this is set to half the inner loop in all integrators that use a barostat
    expq, expp = (np.exp(v * halfdt), np.exp(-v * halfdt))

    baro.nm.qnm[0, :] *= expq
    baro.nm.pnm[0, :] *= expp
    baro.cell.h *= expq


def _pbaro(baro):
    """Propagation step for the cell momentum (adjusting atomic positions and momenta)."""

    # we are assuming then that p the coupling between p^2 and dp/dt only involves the fast force
    dt = baro.pdt[0]

    # computes the pressure associated with the forces at the outer level MTS level.
    press = np.trace(baro.stress_mts(0)) / 3.0
    # integerates the kinetic part of the pressure with the force at the inner-most level.
    nbeads = baro.beads.nbeads
    baro.p += (
        3.0
        * dt
        * (baro.cell.V * (press - nbeads * baro.pext) + Constants.kb * baro.temp)
    )


def get_npt_stepper(
    sim,
    model,
    device,
    rescale_energy=True,
    random_rotation=False,
    use_standard_vv=False,
):
    motion = sim.syslist[0].motion
    if type(motion.integrator) is not NPTIntegrator:
        raise TypeError(
            f"Base i-PI integrator is of type {motion.integrator.__class__.__name__}, use a NPT setup."
        )

    if use_standard_vv:
        # use the standard velocity Verlet integrator
        vv_step = get_standard_vv_step(
            sim, model, device, rescale_energy, random_rotation
        )
    else:
        # defaults to the FlashMD VV stepper
        vv_step = get_flashmd_vv_step(
            sim, model, device, rescale_energy, random_rotation
        )

    # The barostat here needs a simpler splitting than for BZP, something as
    # OAbBbBABbAbPO where Bp and Ap are the cell momentum and volume steps
    def npt_stepper(motion, *_, **__):
        info("@flashmd: Starting NPT step", verbosity.debug)
        info("@flashmd: Particle thermo", verbosity.debug)
        motion.thermostat.step()
        info("@flashmd: P constraints", verbosity.debug)
        motion.integrator.pconstraints()
        info("@flashmd: Barostat thermo", verbosity.debug)
        motion.barostat.thermostat.step()
        info("@flashmd: Barostat q", verbosity.debug)
        _qbaro(motion.barostat)
        info("@flashmd: Barostat p", verbosity.debug)
        _pbaro(motion.barostat)
        info("@flashmd: FlashVV", verbosity.debug)
        vv_step(motion)
        info("@flashmd: Barostat p", verbosity.debug)
        _pbaro(motion.barostat)
        info("@flashmd: Barostat q", verbosity.debug)
        _qbaro(motion.barostat)
        info("@flashmd: Barostat thermo", verbosity.debug)
        motion.barostat.thermostat.step()
        info("@flashmd: Particle thermo", verbosity.debug)
        motion.thermostat.step()
        info("@flashmd: P constraints", verbosity.debug)
        motion.integrator.pconstraints()
        motion.ensemble.time += motion.dt
        info("@flashmd: NPT Step finished", verbosity.debug)

    return npt_stepper


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
        system.positions.cpu().numpy().flatten() * ase.units.Angstrom / ase.units.Bohr
    )
    motion.beads.p[:] = system.get_data("momenta").block().values.squeeze(
        -1
    ).cpu().numpy().flatten() / (
        (9.1093819e-31 * ase.units.kg)
        * (ase.units.Bohr / ase.units.Angstrom)
        / (2.4188843e-17 * ase.units.s)
    )
