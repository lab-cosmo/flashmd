import ase.units
import torch
import vesin.metatomic
from metatomic.torch import AtomisticModel, ModelEvaluationOptions, ModelOutput, System

from ..constraints import enforce_physical_constraints
from . import AtomisticStepper
from .utils import build_system


class FlashMDStepper(AtomisticStepper):
    def __init__(
        self,
        model: AtomisticModel,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.time_step = float(model.module.timestep) * ase.units.fs

        self.evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                "positions": ModelOutput(per_atom=True),
                "momenta": ModelOutput(per_atom=True),
            },
        )

        self.dtype = getattr(torch, self.model.capabilities().dtype)
        self.device = device

    def get_timestep(self) -> float:
        return self.time_step

    def step(self, system: System):
        if system.device.type != self.device.type:
            raise ValueError("System device does not match stepper device.")
        if system.positions.dtype != self.dtype:
            raise ValueError("System dtype does not match stepper dtype.")

        vesin.metatomic.compute_requested_neighbors([system], "angstrom", self.model)

        model_outputs = self.model(
            [system], self.evaluation_options, check_consistency=False
        )
        model_outputs = enforce_physical_constraints(
            [system], model_outputs, timestep=self.time_step
        )

        new_q = model_outputs["positions"].block().values.squeeze(-1)
        new_p = model_outputs["momenta"].block().values.squeeze(-1)

        return build_system(system, new_q, new_p)
