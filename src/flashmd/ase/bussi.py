from .velocity_verlet import VelocityVerlet
import ase.units
from typing import List
from metatomic.torch import AtomisticModel
import torch
import ase
import numpy as np


class Bussi(VelocityVerlet):
    def __init__(
        self,
        atoms: ase.Atoms,
        timestep: float,
        temperature_K: float,
        model: AtomisticModel | List[AtomisticModel],
        time_constant: float = 10.0 * ase.units.fs,
        device: str | torch.device = "auto",
        rescale_energy: bool = True,
        **kwargs,
    ):
        super().__init__(atoms, timestep, model, device, rescale_energy, **kwargs)

        self.temperature_K = temperature_K
        self.time_constant = time_constant

    def step(self):
        self.apply_bussi_half_step()
        super().step()
        self.apply_bussi_half_step()

    def apply_bussi_half_step(self):
        old_kinetic_energy = self.atoms.get_kinetic_energy()
        n_degrees_of_freedom = 3 * len(self.atoms)
        target_kinetic_energy = (
            0.5 * ase.units.kB * self.temperature_K * n_degrees_of_freedom
        )

        exp_term = np.exp(-0.5 * self.dt / self.time_constant)
        energy_scaling_term = (
            (1.0 - exp_term)
            * target_kinetic_energy
            / old_kinetic_energy
            / n_degrees_of_freedom
        )
        r = np.random.randn(n_degrees_of_freedom)
        alpha_sq = (
            exp_term
            + energy_scaling_term * np.sum(r**2)
            + 2.0 * r[0] * np.sqrt(exp_term * energy_scaling_term)
        )
        alpha = np.sqrt(alpha_sq)

        momenta = self.atoms.get_momenta()
        self.atoms.set_momenta(alpha * momenta)
