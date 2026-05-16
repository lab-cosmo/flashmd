# from ..utils.pretrained import load_pretrained_models
import ase.units
import torch
import vesin.metatomic
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import AtomisticModel, ModelEvaluationOptions, ModelOutput, System

from .constraints import enforce_physical_constraints


class FlashMDStepper:
    def __init__(
        self,
        model: AtomisticModel,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.time_step = float(model.module.timestep) * ase.units.fs

        # one of these for each model:
        self.evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                "positions": ModelOutput(per_atom=True),
                "momenta": ModelOutput(per_atom=True),
            },
        )

        self.dtype = getattr(torch, self.model.capabilities().dtype)
        self.device = device
        self.neighbor_list_calculators = vesin.metatomic.neighbor_lists_for_model(
            "angstrom", self.model
        )

        # vesin's CUDA brute_force algorithm is broken for triclinic cells,
        # so we use cell_list for all CUDA calculations to avoid potential issues. See
        # https://github.com/Luthaf/vesin/issues/157
        for neighbor_list in self.neighbor_list_calculators:
            neighbor_list._nl.algorithm = "cell_list"

    def step(self, system: System):
        if system.device.type != self.device.type:
            raise ValueError("System device does not match stepper device.")
        if system.positions.dtype != self.dtype:
            raise ValueError("System dtype does not match stepper dtype.")

        for calculator in self.neighbor_list_calculators:
            calculator.add_neighbor_list(system)

        masses = system.get_data("masses").block().values
        model_outputs = self.model(
            [system], self.evaluation_options, check_consistency=False
        )
        model_outputs = enforce_physical_constraints(
            [system], model_outputs, timestep=self.time_step
        )

        new_q = model_outputs["positions"].block().values.squeeze(-1)
        new_p = model_outputs["momenta"].block().values.squeeze(-1)

        new_system = System(
            positions=new_q,
            types=system.types,
            cell=system.cell,
            pbc=system.pbc,
        )
        new_system.add_data(
            "momenta",
            TensorMap(
                keys=Labels.single().to(self.device),
                blocks=[
                    TensorBlock(
                        values=new_p.unsqueeze(-1),
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor(
                                [[0, j] for j in range(len(new_system))],
                                device=self.device,
                            ),
                        ),
                        components=[
                            Labels(
                                names="xyz",
                                values=torch.tensor(
                                    [[0], [1], [2]], device=self.device
                                ),
                            )
                        ],
                        properties=Labels.single().to(self.device),
                    )
                ],
            ),
        )
        new_system.add_data(
            "masses",
            TensorMap(
                keys=Labels.single().to(self.device),
                blocks=[
                    TensorBlock(
                        values=masses,
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor(
                                [[0, j] for j in range(len(new_system))],
                                device=self.device,
                            ),
                        ),
                        components=[],
                        properties=Labels.single().to(self.device),
                    )
                ],
            ),
        )
        return new_system
