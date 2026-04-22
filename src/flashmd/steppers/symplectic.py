from functools import partial
from typing import Callable

import ase.units
import torch
from metatomic.torch import AtomisticModel, ModelEvaluationOptions, ModelOutput, System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from flashmd.steppers import AtomisticStepper
from flashmd.utils import system_from_template


def system_to_phase_space(system: System) -> torch.Tensor:
    """Flatten a System into a phase-space vector [positions; momenta].

    The flat representation is required because the fixed-point solver operates
    on plain tensors rather than System objects.
    """
    positions = system.positions
    momenta = system.get_data("momenta")[0].values
    return torch.cat([positions.view(-1), momenta.view(-1)], dim=0)


def phase_space_to_system(system: System, x: torch.Tensor) -> System:
    """Reconstruct a System from a flat phase-space vector.

    Inverse of system_to_phase_space. Types, cell, pbc, and masses are copied
    from the template system; positions and momenta are taken from x. This thin
    wrapper exists because the fixed-point solver works on plain tensors, but
    the model inside the loop requires a System object.
    """
    positions, momenta = torch.chunk(x, 2)
    positions = positions.view_as(system.positions)
    momenta = momenta.view_as(system.positions)
    return system_from_template(system, positions, momenta)


class SymplecticStepper(AtomisticStepper):
    def __init__(
        self,
        initial_guess: AtomisticStepper,
        model: AtomisticModel,
        fixed_point_solver: Callable[
            [Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor
        ],
    ):
        """
        Args:
            initial_guess: The stepper to generate the initial guess for the fixed-point
                iterations.
            model: The AtomisticModel that will be used inside the fixed-point
                iterations to compute the updates. The model should take in a midpoint
                system and output the corresponding deltas.
            fixed_point_solver: The function that will be used to solve for the fixed
                point. It should take in a function that computes the update given the
                current guess, and an initial guess for the midpoint, and return the
                converged midpoint.
        """
        self.initial_guess = initial_guess
        self.model = model
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
        return float(self.model.module.timestep) * ase.units.fs

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
        # convert to system representation
        midpoint_system = phase_space_to_system(system, x_bar)

        # attach neighbor lists based on the model's requests
        midpoint_system = get_system_with_neighbor_lists(
            midpoint_system, self.model.requested_neighbor_lists()
        )

        # run the model to get the deltas
        outputs = self.model(
            [midpoint_system], self.evaluation_options, check_consistency=False
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
