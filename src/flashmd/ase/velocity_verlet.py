from ase.md.md import MolecularDynamics
from typing import List
from metatensor.torch.atomistic import MetatensorAtomisticModel
from metatensor.torch import Labels, TensorBlock, TensorMap
import ase.units
import torch
from metatensor.torch.atomistic.ase_calculator import _ase_to_torch_data
from metatensor.torch.atomistic import System
import ase
from ..stepper import FlashMDStepper

import numpy as np
from scipy.spatial.transform import Rotation
class VelocityVerlet(MolecularDynamics):
    def __init__(
        self,
        atoms: ase.Atoms,
        timestep: float,
        model: MetatensorAtomisticModel | List[MetatensorAtomisticModel],
        device: str | torch.device = "auto",
        rescale_energy: bool = True,
        random_rotation: bool = False,
        **kwargs,
    ):
        super().__init__(atoms, timestep, **kwargs)

        models = model if isinstance(model, list) else [model]
        capabilities = models[0].capabilities()

        base_timestep = float(models[0].module.base_time_step) * ase.units.fs

        n_time_steps = int(
            [k for k in capabilities.outputs.keys() if "mtt::delta_" in k][0].split(
                "_"
            )[1]
        )
        if n_time_steps != self.dt / base_timestep:
            raise ValueError(
                f"Mismatch between timestep ({self.dt}) and model timestep ({base_timestep})."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.device = torch.device(device)
        self.dtype = getattr(torch, capabilities.dtype)

        self.stepper = FlashMDStepper(models, n_time_steps, self.device)
        self.rescale_energy = rescale_energy
        self.random_rotation = random_rotation

    def step(self):

        if self.rescale_energy:
            old_energy = self.atoms.get_total_energy()

        system = _convert_atoms_to_system(
            self.atoms, device=self.device, dtype=self.dtype
        )

        if self.random_rotation:
            # generate a random rotation matrix (with i-PI utils for consistency)
            R = torch.tensor(
                _random_R(),
                device=system.positions.device,
                dtype=system.positions.dtype,
            )
            # apply the random rotation
            system.cell = system.cell @ R.T
            system.positions = system.positions @ R.T
            momenta = system.get_data("momenta").block(0).values.squeeze()
            momenta[:] = momenta @ R.T  # does the change in place

        new_system = self.stepper.step(system)

        if self.random_rotation:
            # revert q, p, and cell to the original reference frame
            new_system.cell = system.cell @ R
            new_system.positions = system.positions @ R
            new_momenta = new_system.get_data("momenta").block(0).values.squeeze()
            new_momenta[:] = new_momenta @ R

        self.atoms.set_positions(new_system.positions.detach().cpu().numpy())
        self.atoms.set_momenta(
            new_system.get_data("momenta")
            .block()
            .values.squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )

        if self.rescale_energy:
            new_energy = self.atoms.get_total_energy()
            old_kinetic_energy = self.atoms.get_kinetic_energy()
            alpha = np.sqrt(1.0 - (new_energy - old_energy) / old_kinetic_energy)
            self.atoms.set_momenta(alpha * self.atoms.get_momenta())


def _convert_atoms_to_system(
    atoms: ase.Atoms, dtype: str, device: str | torch.device
) -> System:
    system_data = _ase_to_torch_data(atoms, dtype=dtype, device=device)
    system = System(*system_data)
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=torch.tensor(
                        atoms.get_momenta(), dtype=dtype, device=device
                    ).unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(len(atoms))], device=device
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
                    values=torch.tensor(
                        atoms.get_masses(), dtype=dtype, device=device
                    ).unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(len(atoms))], device=device
                        ),
                    ),
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )
    return system


def _random_R():
    R = Rotation.random().as_matrix()
    if np.random.rand() < 0.5:
        R[:, 0] *= -1
    return R
