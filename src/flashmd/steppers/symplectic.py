from functools import partial
from typing import Callable

import ase.units
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import AtomisticModel, ModelEvaluationOptions, ModelOutput, System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from flashmd.steppers import AtomisticStepper


def system_to_phase_space(system) -> torch.Tensor:
    # extract positions and momenta from system
    positions = system.positions
    momenta = system.get_data("momenta")[0].values
    # flatten and concatenate
    return torch.cat([positions.view(-1), momenta.view(-1)], dim=0)


def phase_space_to_system(system, x: torch.Tensor):
    # extract positions and momenta from concatenated tensor and reshape into original shapes
    positions, momenta = torch.chunk(x, 2)
    positions = positions.view_as(system.positions)
    momenta = momenta.view_as(system.get_data("momenta")[0].values)

    # take the types, masses and cell from the original system
    new_system = System(
        types=system.types,
        positions=positions,
        cell=system.cell,
        pbc=system.pbc,
    )

    # copy masses
    new_system.add_data("masses", system.get_data("masses"))

    # attach momenta
    device = positions.device
    new_system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=momenta,
                    samples=Labels.range("atom", len(system)).to(device),
                    components=[Labels.range("xyz", 3).to(device)],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )

    return new_system


class SymplecticStepper(AtomisticStepper):
    def __init__(
        self,
        initial_guess: AtomisticStepper,
        midpoint_to_delta_model: AtomisticModel,
        fixed_point_solver: Callable[
            [Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor
        ],
    ):
        # super().__init__(flashmd, device)
        self.initial_guess = initial_guess
        self.midpoint_to_delta_model = midpoint_to_delta_model
        self.fixed_point_solver = fixed_point_solver

        # self.model = model
        self.evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                "positions": ModelOutput(per_atom=True),
                "momenta": ModelOutput(per_atom=True),
            },
        )
        self.fixed_point_solver = fixed_point_solver

    def get_timestep(self) -> float:
        timestep: float = self.midpoint_to_delta_model.module.timestep.item()  # type: ignore
        return timestep * ase.units.fs

    def _fixed_point_step(
        self, system, x_init: torch.Tensor, x_bar: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the current estimate of the midpoint in phase-space representation, update and
        return it.

        NOTE: The function takes a system as the first argument to allow constructing a
        metatomic-compatible System object, which unfortunately is required for model
        evaluation.

        Args:
            system: The initial system before the step.
            x_init: The initial system in phase-space representation. For the fixed-point
                iterations, it has to be of shape (B, D) where B is the batch size (1 here) and
                D is the dimension of the phase space.
            x_bar: The current estimate of the midpoint in phase-space representation. Note
                that this also has to be of shape (B, D).

        Returns:
            The updated midpoint in phase-space representation.
        """
        # flatten the batch dimension
        x_bar = x_bar.squeeze(0)

        # convert to system representation
        midpoint_system = phase_space_to_system(system, x_bar)

        # attach neighbor lists based on the model's requests
        midpoint_system = get_system_with_neighbor_lists(
            midpoint_system, self.midpoint_to_delta_model.requested_neighbor_lists()
        )

        # run the model to get the deltas
        evaluation_options = self.evaluation_options
        outputs = self.midpoint_to_delta_model(
            [midpoint_system], evaluation_options, check_consistency=False
        )

        # depending on the model, extract deltas
        delta_q = outputs["positions"].block().values.squeeze(-1)
        delta_p = outputs["momenta"].block().values

        # compute new midpoint in phase space
        delta_x = torch.cat([delta_q.view(-1), delta_p.view(-1)], dim=0)

        # compute new midpoint
        x_bar_new = x_init + 0.5 * delta_x
        return x_bar_new

    def step(self, system: System) -> System:  # type: ignore
        # convert system to phase space representation
        x_init = system_to_phase_space(system)

        # get initial guess from FlashMD
        initial_guess = self.initial_guess.step(system)
        x_prime_init = system_to_phase_space(initial_guess)

        # compute initial midpoint from starting point and initial guess
        x_bar_init = 0.5 * (x_init + x_prime_init)

        # attach the system to the fixed-point function and call solver
        f = partial(self._fixed_point_step, system, x_init)
        x_bar_star = self.fixed_point_solver(f, x_bar_init)

        # compute final updated phase space point
        x_star = 2 * x_bar_star - x_init

        # convert back to system representation
        x_prime = phase_space_to_system(system, x_star)

        return x_prime
