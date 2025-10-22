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


class VelocityVerlet(MolecularDynamics):
    def __init__(
        self,
        atoms: ase.Atoms,
        timestep: float,
        model: AtomisticModel | List[AtomisticModel],
        device: str | torch.device = "auto",
        rescale_energy: bool = True,
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

    def step(self):
        if self.rescale_energy:
            old_energy = self.atoms.get_total_energy()

        system = _convert_atoms_to_system(
            self.atoms, device=self.device, dtype=self.dtype
        )
        new_system = self.stepper.step(system)
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
