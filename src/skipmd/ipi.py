from ipi.utils.depend import dstrip
from ipi.utils.mathtools import (
    random_rotation,
)
from ipi.engine.cell import GenericCell

from skipmd.stepper import SkipMDStepper
import ase.units
import torch
import numpy as np
import ase.data

from metatensor.torch.atomistic import System
from metatensor.torch import Labels, TensorBlock, TensorMap


def get_skipmd_velocity_verlet_step(sim, model, device):

    capabilities = model.capabilities()

    base_timestep = float(model.module.base_time_step) * ase.units.fs

    dt = sim.simulation.syslist[0].motion.dt * 2.4188843e-17 * ase.units.s

    n_time_steps = int([k for k in capabilities.outputs.keys() if "mtt::delta_" in k][0].split("_")[1])
    if not np.allclose(dt, n_time_steps * base_timestep):
        raise ValueError(
            f"Mismatch between timestep ({dt}) and model timestep ({base_timestep})."
        )

    device = torch.device(device)
    dtype = getattr(torch, capabilities.dtype)
    stepper = SkipMDStepper([model], n_time_steps, device)

    def skipmd_vv(motion, rescale_energy=True, rand_rot=False):

        if rescale_energy:
            old_energy = sim.properties("potential") + sim.properties("kinetic_md")

        if rand_rot:
            # OBTAIN A RANDOM ROTATION
            R = random_rotation(self.prng, improper=True)

            # APPLY RANDOM ROTATION TO SYSTEM
            rot_motion = motion.clone()
            rot_motion.cell.h = GenericCell(
                R @ dstrip(rot_motion.cell.h).copy()
            )
            rot_motion.beads.q[:] = (dstrip(rot_motion.beads.q).reshape(-1, 3) @ R.T).flatten()
            rot_motion.beads.p[:] = (dstrip(rot_motion.beads.p).reshape(-1, 3) @ R.T).flatten()
            system = ipi_to_system(rot_motion, device, dtype)
        else:
            system = ipi_to_system(motion, device, dtype)

        new_system = stepper.step(system)

        if rand_rot:
            system_to_ipi(rot_motion, new_system)
            # UNDO THE RANDOM ROTATION ON Q AND P, WRITE TO "MAIN" MOTION
            motion.beads.q[:] = (dstrip(rot_motion.beads.q).reshape(-1, 3) @ R).flatten()
            motion.beads.p[:] = (dstrip(rot_motion.beads.p).reshape(-1, 3) @ R).flatten()

        else:
            system_to_ipi(motion, new_system)

        motion.integrator.pconstraints()

        if rescale_energy:
            new_energy = sim.properties("potential") + sim.properties("kinetic_md")
            old_kinetic_energy = sim.properties("kinetic_md")
            alpha = np.sqrt(1.0 - (new_energy - old_energy) / old_kinetic_energy)
            motion.beads.p[:] = alpha * dstrip(motion.beads.p)

    return skipmd_vv


def ipi_to_system(motion, device, dtype):
    positions = dstrip(motion.beads.q).reshape(-1, 3) * ase.units.Bohr / ase.units.Angstrom
    positions_torch = torch.tensor(positions, device=device, dtype=dtype)
    cell = dstrip(motion.cell.h) * ase.units.Bohr / ase.units.Angstrom
    cell_torch = torch.tensor(cell, device=device, dtype=dtype)
    pbc_torch = torch.tensor([True, True, True], device=device, dtype=torch.bool)
    momenta = dstrip(motion.beads.p).reshape(-1, 3) * (9.1093819e-31 * ase.units.kg) * (ase.units.Bohr / ase.units.Angstrom) / (2.4188843e-17 * ase.units.s)
    momenta_torch = torch.tensor(momenta, device=device, dtype=dtype)
    masses = dstrip(motion.beads.m) * 9.1093819e-31 * ase.units.kg
    masses_torch = torch.tensor(masses, device=device, dtype=dtype)
    types_torch = torch.tensor([ase.data.atomic_numbers[name] for name in motion.beads.names], device=device, dtype=torch.int32)
    system = System(types_torch, positions_torch, cell_torch, pbc_torch)
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks = [
                TensorBlock(
                    values=momenta_torch.unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, j] for j in range(len(momenta_torch))], device=device),
                    ),
                    components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=device))],
                    properties=Labels.single().to(device),
                )
            ],
        )
    )
    system.add_data(
        "masses",
        TensorMap(
            keys=Labels.single().to(device),
            blocks = [
                TensorBlock(
                    values=masses_torch.unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, j] for j in range(len(masses_torch))], device=device),
                    ),
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        )
    )
    return system

def system_to_ipi(motion, system):
    motion.beads.q[:] = system.positions.cpu().numpy().flatten() * ase.units.Angstrom / ase.units.Bohr
    motion.beads.p[:] = system.get_data("momenta").block().values.squeeze(-1).cpu().numpy().flatten() / ((9.1093819e-31 * ase.units.kg) * (ase.units.Bohr / ase.units.Angstrom) / (2.4188843e-17 * ase.units.s))
