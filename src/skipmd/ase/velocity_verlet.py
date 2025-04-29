from ase.md.md import MolecularDynamics
# from ..utils.pretrained import load_pretrained_models
from metatensor.torch.atomistic import ModelEvaluationOptions, ModelOutput
from metatensor.torch import Labels, TensorBlock, TensorMap
import ase.units
import torch
from metatensor.torch.atomistic.ase_calculator import _ase_to_torch_data, _compute_ase_neighbors
from metatensor.torch.atomistic import System
import numpy as np


class VelocityVerlet(MolecularDynamics):
    def __init__(self, model, base_timestep, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.n_time_steps = int([k for k in model.capabilities().outputs.keys() if "mtt::delta_" in k][0].split("_")[1])

        if self.n_time_steps != self.dt / base_timestep:
            raise ValueError(
                f"Mismatch between timestep ({self.dt}) and model timestep ({base_timestep})."
            )

        self.evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                f"mtt::delta_{self.n_time_steps}_q": ModelOutput(per_atom=True),
                f"mtt::p_{self.n_time_steps}": ModelOutput(per_atom=True),
            }
        )
        self.dtype = torch.float32
        self.device = device

    def step(self):

        system_data = _ase_to_torch_data(self.atoms, dtype=self.dtype, device=self.device)
        system = System(*system_data)
        for options in self.model.requested_neighbor_lists():
            neighbors = _compute_ase_neighbors(
                self.atoms, options, dtype=self.dtype, device=self.device
            )
            system.add_neighbor_list(options, neighbors)
        system.add_data(
            "momenta",
            TensorMap(
                keys=Labels.single().to(self.device),
                blocks = [
                    TensorBlock(
                        values=torch.tensor(self.atoms.get_momenta(), dtype=self.dtype, device=self.device).unsqueeze(-1),
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor([[0, j] for j in range(len(self.atoms))], device=self.device),
                        ),
                        components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=self.device))],
                        properties=Labels.single().to(self.device),
                    )
                ],
            )
        )
        model_outputs = self.model([system], self.evaluation_options, check_consistency=False)
        delta_q_scaled = model_outputs[f"mtt::delta_{self.n_time_steps}_q"].block().values.squeeze(-1).detach().cpu().numpy()
        p_scaled = model_outputs[f"mtt::p_{self.n_time_steps}"].block().values.squeeze(-1).detach().cpu().numpy()
        sqrt_masses = np.sqrt(self.atoms.get_masses()[:, None])
        delta_q = delta_q_scaled / sqrt_masses
        p = p_scaled * sqrt_masses
        self.atoms.set_positions(self.atoms.get_positions() + delta_q)
        self.atoms.set_momenta(p)
