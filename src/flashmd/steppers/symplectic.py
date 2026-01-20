from typing import Callable

import ase.units
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import AtomisticModel, ModelEvaluationOptions, ModelOutput, System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from flashmd.steppers import AtomisticStepper


class SymplecticStepper(AtomisticStepper):
    def __init__(
        self,
        initial_guess: AtomisticStepper,
        midpoint_to_delta_model: AtomisticModel,
        fixed_point_solver: Callable[
            [Callable[[torch.Tensor], torch.Tensor], torch.Tensor], torch.Tensor
        ],
        # device: torch.device,
        # accuracy_threshold: float = 1e-3,
        # alpha: float = 0.5,
    ):
        # super().__init__(flashmd, device)
        self.initial_guess = initial_guess
        self.midpoint_to_delta_model = midpoint_to_delta_model
        self.fixed_point_solver = fixed_point_solver

        # self.model = model
        self.evaluation_options_implicit = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                "positions": ModelOutput(per_atom=True),
                "momenta": ModelOutput(per_atom=True),
            },
        )
        self.accuracy_threshold = 1e-3
        self.alpha = 0.5

    def get_timestep(self) -> float:
        timestep: float = self.midpoint_to_delta_model.module.timestep.item()  # type: ignore
        return timestep * ase.units.fs

    def step(self, system: System) -> System:  # type: ignore
        new_system = self.initial_guess.step(system)
        # new_system = system

        cooldown = 300
        accuracy = np.inf
        accuracies = [np.inf]
        accuracy_threshold = self.accuracy_threshold
        alpha = self.alpha
        niterations = 0
        old_positions = new_system.positions
        old_momenta = new_system.get_data("momenta").block().values
        while accuracy > accuracy_threshold:
            print("Iteration:", niterations, "Accuracy:", accuracy)
            old_positions = new_system.positions * alpha + old_positions * (1 - alpha)
            old_momenta = new_system.get_data(
                "momenta"
            ).block().values * alpha + old_momenta * (1 - alpha)
            midpoint_system = get_system(
                (system.positions + old_positions) / 2.0,
                system.types,
                system.cell,
                system.pbc,
                (system.get_data("momenta").block().values + old_momenta) / 2.0,
                system.get_data("masses").block().values,
            )
            midpoint_system = get_system_with_neighbor_lists(
                midpoint_system, self.midpoint_to_delta_model.requested_neighbor_lists()
            )
            outputs = self.midpoint_to_delta_model(
                [midpoint_system],
                self.evaluation_options_implicit,
                check_consistency=False,
            )
            delta_q = outputs["positions"].block().values.squeeze(-1)
            delta_p = outputs["momenta"].block().values
            new_system = get_system(
                system.positions + delta_q,
                system.types,
                system.cell,
                system.pbc,
                system.get_data("momenta").block().values + delta_p,
                system.get_data("masses").block().values,
            )
            accuracy = (
                torch.abs(new_system.positions - old_positions).max().item()
                + torch.abs(new_system.get_data("momenta").block().values - old_momenta)
                .max()
                .item()
            )
            # print(torch.abs(new_system.positions - old_positions).max().item(), torch.abs(new_system.get_data("momenta").block().values - old_momenta).max().item())
            accuracies.append(accuracy)
            if len(accuracies) > 100:
                if accuracy > accuracies[-100] and cooldown <= 0:
                    print("Reducing alpha")
                    alpha *= 0.5
                    cooldown = 300
            niterations += 1
            cooldown -= 1
        print(
            "Number of iterations:",
            niterations,
            "accuracy threshold:",
            accuracy_threshold,
        )
        return new_system


def get_system(positions, types, cell, pbc, momenta, masses):
    device = positions.device
    system = System(
        positions=positions,
        types=types,
        cell=cell,
        pbc=pbc,
    )
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=momenta,
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(len(system))],
                            device=device,
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
                    values=masses,
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor(
                            [[0, j] for j in range(len(system))],
                            device=device,
                        ),
                    ),
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )
    return system
