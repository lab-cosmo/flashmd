# from ..utils.pretrained import load_pretrained_models
from metatomic.torch import ModelEvaluationOptions, ModelOutput
from metatensor.torch import Labels, TensorBlock, TensorMap
import torch
from metatomic.torch import System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatomic.torch import AtomisticModel


class FlashMDStepper:
    def __init__(
        self,
        model: AtomisticModel,
        device: torch.device,
    ):
        self.model = model.to(device)

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

    def step(self, system: System):
        if system.device.type != self.device.type:
            raise ValueError("System device does not match stepper device.")
        if system.positions.dtype != self.dtype:
            raise ValueError("System dtype does not match stepper dtype.")

        system = get_system_with_neighbor_lists(
            system, self.model.requested_neighbor_lists()
        )

        masses = system.get_data("masses").block().values
        model_outputs = self.model(
            [system], self.evaluation_options, check_consistency=False
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
