import torch
from metatomic.torch import System

from ..utils import make_system


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
    masses = template.get_data("masses").block().values.squeeze(-1)
    return make_system(
        template.types, positions, template.cell, template.pbc, momenta, masses
    )
