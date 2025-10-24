from ase.md.md import MolecularDynamics
from typing import List
from metatomic.torch import AtomisticModel
from metatensor.torch import Labels, TensorBlock, TensorMap
import ase.units
import torch
from metatomic.torch.ase_calculator import _ase_to_torch_data
from metatomic.torch import System
import ase
from ..stepper import FlashMDStepper
import numpy as np
from scipy.spatial.transform import Rotation


class VelocityVerlet(MolecularDynamics):
    def __init__(
        self,
        atoms: ase.Atoms,
        timestep: float,
        model: AtomisticModel | List[AtomisticModel],
        device: str | torch.device = "auto",
        rescale_energy: bool = True,
        random_rotation: bool = False,
        **kwargs,
    ):
        super().__init__(atoms, timestep, **kwargs)

        capabilities = model.capabilities()

        model_timestep = float(model.module.timestep)
        if not np.allclose(model_timestep, self.dt / ase.units.fs):
            raise ValueError(
                f"Mismatch between timestep ({self.dt / ase.units.fs} fs) "
                f"and model timestep ({model_timestep} fs)."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device
        self.device = torch.device(device)
        self.dtype = getattr(torch, capabilities.dtype)

        self.stepper = FlashMDStepper(model, self.device)
        self.rescale_energy = rescale_energy
        self.random_rotation = random_rotation

    def step(self):
        if self.rescale_energy:
            old_energy = self.atoms.get_total_energy()

        system = _convert_atoms_to_system(
            self.atoms, device=self.device, dtype=self.dtype
        )

        if self.random_rotation:
            # generate a random rotation matrix with SciPy
            R = torch.tensor(
                _get_random_rotation(),
                device=system.positions.device,
                dtype=system.positions.dtype,
            )
            # apply the random rotation
            old_cell = system.cell
            system.cell = system.cell @ R.T
            system.positions = system.positions @ R.T
            # change momentum TensorMap in place
            system.get_data("momenta").block().values[:] = (
                (system.get_data("momenta").block().values.squeeze(-1) @ R.T).unsqueeze(-1)
            )

        new_system = self.stepper.step(system)

        if self.random_rotation:
            # revert q, p to the original reference frame, load old cell
            new_system.cell = old_cell
            new_system.positions = new_system.positions @ R
            new_system.get_data("momenta").block().values[:] = (
                (new_system.get_data("momenta").block().values.squeeze(-1) @ R).unsqueeze(-1)
            )

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


def _get_random_rotation():
    R = Rotation.random().as_matrix()
    if np.random.rand() < 0.5:
        R *= -1  # allow improper rotations
    return R
