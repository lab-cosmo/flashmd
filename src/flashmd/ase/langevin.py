from .velocity_verlet import VelocityVerlet
import ase.units
from typing import List
from metatomic.torch import AtomisticModel
import torch
import ase
import numpy as np


class Langevin(VelocityVerlet):
    def __init__(
        self,
        atoms: ase.Atoms,
        timestep: float,
        temperature_K: float,
        model: AtomisticModel | List[AtomisticModel],
        time_constant: float = 100.0 * ase.units.fs,
        fixcm: bool = True,
        device: str | torch.device = "auto",
        rescale_energy: bool = False,
        random_rotation: bool = False,
        **kwargs,
    ):
        super().__init__(atoms, timestep, model, device, rescale_energy, random_rotation, **kwargs)

        self.temperature_K = temperature_K
        self.friction = 1.0 / time_constant
        self.fixcm = fixcm
        if self.fixcm:
            self.atoms.set_velocities(self.atoms.get_velocities() - self.atoms.get_velocities().mean(axis=0))

    def step(self):
        self.apply_langevin_half_step()
        super().step()
        self.apply_langevin_half_step()

    def apply_langevin_half_step(self):
        old_momenta = self.atoms.get_momenta()
        new_momenta = np.exp(-self.friction * 0.5 * self.dt) * old_momenta + np.sqrt(
            1.0 - np.exp(-self.friction * self.dt)
        ) * np.sqrt(
            ase.units.kB * self.temperature_K * self.atoms.get_masses()[:, None]
        ) * np.random.randn(*old_momenta.shape)
        self.atoms.set_momenta(new_momenta)
        if self.fixcm:
            self.atoms.set_velocities(self.atoms.get_velocities() - self.atoms.get_velocities().mean(axis=0))
