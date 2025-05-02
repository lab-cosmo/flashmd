from ase.md.md import MolecularDynamics
# from ..utils.pretrained import load_pretrained_models
from metatensor.torch.atomistic import ModelEvaluationOptions, ModelOutput
from metatensor.torch import Labels, TensorBlock, TensorMap
import ase.units
import torch
from metatensor.torch.atomistic.ase_calculator import _ase_to_torch_data, _compute_ase_neighbors
from metatensor.torch.atomistic import System
import numpy as np
import torch
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from typing import List, Optional
from metatensor.torch.atomistic import MetatensorAtomisticModel


class SkipMDStepper:
    def __init__(
        self,
        models: List[MetatensorAtomisticModel],
        n_time_steps: int,
        device: torch.device,
        energy_model: Optional[MetatensorAtomisticModel],
        # q_error_threshold: float = 0.1,
        # p_error_threshold: float = 0.1,
        # energy_error_threshold: float = 0.1,
    ):

        self.n_time_steps = n_time_steps

        # internally, turn list of models into a dict and send to device
        self.models = {}
        for model in models:
            n_time_steps_model = int([k for k in model.capabilities().outputs.keys() if "mtt::delta_" in k][0].split("_")[1])
            self.models[n_time_steps_model] = model.to(device)

        # one of these for each model:
        self.evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                f"mtt::delta_{self.n_time_steps}_q": ModelOutput(per_atom=True),
                f"mtt::p_{self.n_time_steps}": ModelOutput(per_atom=True),
            }
        )

        self.dtype = getattr(torch, self.models[n_time_steps].capabilities().dtype)
        self.device = device

    def step(self, system: System):

        if system.device.type != self.device.type:
            raise ValueError("System device does not match stepper device.")
        if system.positions.dtype != self.dtype:
            raise ValueError("System dtype does not match stepper dtype.")

        system = get_system_with_neighbor_lists(system, self.models[self.n_time_steps].requested_neighbor_lists())

        masses = system.get_data("masses").block().values
        model_outputs = self.models[self.n_time_steps]([system], self.evaluation_options, check_consistency=False)
        delta_q_scaled = model_outputs[f"mtt::delta_{self.n_time_steps}_q"].block().values.squeeze(-1)
        p_scaled = model_outputs[f"mtt::p_{self.n_time_steps}"].block().values.squeeze(-1)
        sqrt_masses = torch.sqrt(masses)
        delta_q = delta_q_scaled / sqrt_masses
        p = p_scaled * sqrt_masses

        new_system = System(
            positions=system.positions + delta_q,
            types=system.types,
            cell=system.cell,
            pbc=system.pbc,
        )
        new_system.add_data(
            "momenta",
            TensorMap(
                keys=Labels.single().to(self.device),
                blocks = [
                    TensorBlock(
                        values=p.unsqueeze(-1),
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor([[0, j] for j in range(len(new_system))], device=self.device),
                        ),
                        components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=self.device))],
                        properties=Labels.single().to(self.device),
                    )
                ],
            )
        )
        new_system.add_data(
            "masses",
            TensorMap(
                keys=Labels.single().to(self.device),
                blocks = [
                    TensorBlock(
                        values=masses,
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor([[0, j] for j in range(len(new_system))], device=self.device),
                        ),
                        components=[],
                        properties=Labels.single().to(self.device),
                    )
                ],
            )
        )
        return new_system


