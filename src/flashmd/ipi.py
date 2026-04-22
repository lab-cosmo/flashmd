"""i-PI integration helpers for FlashMD.

VV step builders return a callable ``vv_step(motion)`` suitable for passing
to the ensemble wrappers in ``flashmd.wrappers``:

- ``get_vv_step_from_ipi``: uses i-PI's own force evaluations (no ML model).
- ``get_vv_step_from_stepper``: drives a pre-built ``AtomisticStepper`` (e.g.
  ``SymplecticStepper``).
- ``get_vv_step_from_model``: convenience wrapper that builds a
  ``FlashMDStepper`` from a model and delegates to ``get_vv_step_from_stepper``.

Ensemble steppers — combine a VV step with thermostat/barostat logic and
return a ready-to-use ``stepper(motion)`` for i-PI:

- ``get_nve_stepper``
- ``get_nvt_stepper``
- ``get_npt_stepper``
"""

import ase.data
import ase.units
import numpy as np
import torch
from ipi.utils.depend import dstrip
from ipi.utils.mathtools import random_rotation as random_rotation_matrix
from ipi.utils.messages import info, verbosity

from flashmd.steppers import AtomisticStepper
from flashmd.steppers.flashmd import FlashMDStepper
from flashmd.utils import system_from_parts
from flashmd.wrappers import wrap_npt, wrap_nve, wrap_nvt


def get_vv_step_from_ipi(sim, rescale_energy=False, random_rotation=False):
    """Velocity Verlet step using i-PI's own force evaluations."""

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


def get_vv_step_from_stepper(
    sim,
    stepper: AtomisticStepper,
    device: torch.device,
    dtype: torch.dtype,
    rescale_energy: bool = False,
    random_rotation: bool = False,
):
    """Velocity Verlet step driven by a pre-built AtomisticStepper.

    Use this when you need to supply a custom stepper (e.g. SymplecticStepper).
    For the plain FlashMD model case, prefer get_vv_step_from_model.
    """
    model_timestep = stepper.get_timestep()
    dt = sim.syslist[0].motion.dt * 2.4188843e-17 * ase.units.s
    if not np.allclose(dt, model_timestep):
        raise ValueError(
            f"Mismatch between i-PI timestep ({dt}) and model timestep ({model_timestep})."
        )

    def flashmd_vv(motion):
        info("@flashmd: Starting VV", verbosity.debug)
        if rescale_energy:
            info("@flashmd: Old energy", verbosity.debug)
            old_energy = sim.properties("conserved")

        info("@flashmd: Stepper", verbosity.debug)
        system = ipi_to_system(motion, device, dtype)

        if random_rotation:
            R = torch.tensor(
                random_rotation_matrix(motion.prng, improper=True),
                device=system.positions.device,
                dtype=system.positions.dtype,
            )
            system.cell = system.cell @ R.T
            system.positions = system.positions @ R.T
            momenta = system.get_data("momenta").block(0).values.squeeze()
            momenta[:] = momenta @ R.T

        new_system = stepper.step(system)

        if random_rotation:
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
            motion.integrator.pconstraints()

        info("@flashmd: End of VV step", verbosity.debug)

    return flashmd_vv


def get_vv_step_from_model(
    sim, model, device, rescale_energy=False, random_rotation=False
):
    """Velocity Verlet step built from a FlashMD model."""
    capabilities = model.capabilities()
    device = torch.device(device)
    dtype = getattr(torch, capabilities.dtype)
    stepper = FlashMDStepper(model, device)
    return get_vv_step_from_stepper(
        sim, stepper, device, dtype, rescale_energy, random_rotation
    )


def get_nve_stepper(
    sim,
    model,
    device,
    rescale_energy=True,
    random_rotation=False,
    use_standard_vv=False,
):
    """NVE stepper combining a VV step with time propagation."""
    if use_standard_vv:
        vv_step = get_vv_step_from_ipi(sim, rescale_energy, random_rotation)
    else:
        vv_step = get_vv_step_from_model(
            sim, model, device, rescale_energy, random_rotation
        )
    return wrap_nve(sim, vv_step)


def get_nvt_stepper(
    sim,
    model,
    device,
    rescale_energy=False,
    random_rotation=False,
    use_standard_vv=False,
):
    """NVT stepper using an OBABO thermostat splitting around the VV step."""
    if use_standard_vv:
        vv_step = get_vv_step_from_ipi(sim, rescale_energy, random_rotation)
    else:
        vv_step = get_vv_step_from_model(
            sim, model, device, rescale_energy, random_rotation
        )
    return wrap_nvt(sim, vv_step)


def get_npt_stepper(
    sim,
    model,
    device,
    rescale_energy=False,
    random_rotation=False,
    use_standard_vv=False,
):
    """NPT stepper with thermostat and barostat splitting around the VV step."""
    if use_standard_vv:
        vv_step = get_vv_step_from_ipi(sim, rescale_energy, random_rotation)
    else:
        vv_step = get_vv_step_from_model(
            sim, model, device, rescale_energy, random_rotation
        )
    return wrap_npt(sim, vv_step)


def ipi_to_system(motion, device, dtype):
    """Convert an i-PI motion object to a metatomic System."""
    positions = torch.tensor(
        dstrip(motion.beads.q).reshape(-1, 3) * ase.units.Bohr / ase.units.Angstrom,
        device=device,
        dtype=dtype,
    )
    cell = torch.tensor(
        dstrip(motion.cell.h).T * ase.units.Bohr / ase.units.Angstrom,
        device=device,
        dtype=dtype,
    )
    pbc = torch.tensor([True, True, True], device=device, dtype=torch.bool)
    momenta = torch.tensor(
        dstrip(motion.beads.p).reshape(-1, 3)
        * (9.1093819e-31 * ase.units.kg)
        * (ase.units.Bohr / ase.units.Angstrom)
        / (2.4188843e-17 * ase.units.s),
        device=device,
        dtype=dtype,
    )
    masses = torch.tensor(
        dstrip(motion.beads.m) * 9.1093819e-31 * ase.units.kg,
        device=device,
        dtype=dtype,
    )
    types = torch.tensor(
        [ase.data.atomic_numbers[name] for name in motion.beads.names],
        device=device,
        dtype=torch.int32,
    )
    return system_from_parts(types, positions, cell, pbc, momenta, masses)


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
