import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


def make_system(
    types: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    momenta: torch.Tensor,
    masses: torch.Tensor,
) -> System:
    """Build a System from raw tensors.

    Args:
        types: Atomic types, shape (N,).
        positions: Atom positions, shape (N, 3).
        cell: Cell matrix, shape (3, 3).
        pbc: Periodic boundary conditions, shape (3,).
        momenta: Atom momenta, shape (N, 3).
        masses: Atom masses, shape (N,).

    Returns:
        System with momenta and masses attached as TensorMaps.
    """
    device = positions.device
    n_atoms = len(types)

    atom_samples = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, j] for j in range(n_atoms)], device=device),
    )

    system = System(types, positions, cell, pbc)
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=momenta.unsqueeze(-1),
                    samples=atom_samples,
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
                    values=masses.unsqueeze(-1),
                    samples=atom_samples,
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )
    return system
