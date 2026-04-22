import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


def build_system(
    template: System,
    positions: torch.Tensor,
    momenta: torch.Tensor,
) -> System:
    """Build a new System from updated positions and momenta tensors.

    Copies types, cell, pbc, and masses from the template system.

    Args:
        template: Source system for structural data (types, cell, pbc, masses).
        positions: New atom positions, shape (N, 3).
        momenta: New atom momenta, shape (N, 3).

    Returns:
        New System with updated positions and momenta.
    """
    device = positions.device
    n_atoms = len(template)

    new_system = System(
        types=template.types,
        positions=positions,
        cell=template.cell,
        pbc=template.pbc,
    )
    new_system.add_data("masses", template.get_data("masses"))
    new_system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=momenta.unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(n_atoms)], device=device
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
    return new_system
