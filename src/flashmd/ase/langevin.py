import ase
import ase.units
import numpy as np
import torch
from metatomic.torch import AtomisticModel

from ..steppers import AtomisticStepper
from ..steppers.flashmd import FlashMDStepper
from .velocity_verlet import VelocityVerlet


class Langevin(VelocityVerlet):
    def __init__(
        self,
        atoms: ase.Atoms,
        timestep: float,
        temperature_K: float,
        stepper: AtomisticStepper,
        device: torch.device,
        dtype: torch.dtype,
        time_constant: float = 100.0 * ase.units.fs,
        fixcm: bool = True,
        rescale_energy: bool = False,
        random_rotation: bool = False,
        **kwargs,
    ):
        super().__init__(
            atoms, timestep, stepper, device, dtype, rescale_energy, random_rotation, **kwargs
        )

        self.temperature_K = temperature_K
        self.friction = 1.0 / time_constant
        self.fixcm = fixcm
        if self.fixcm:
            self.atoms.set_velocities(
                self.atoms.get_velocities()
                - self.atoms.get_momenta().sum(axis=0) / self.atoms.get_masses().sum()
            )

    @classmethod
    def from_model(
        cls,
        atoms: ase.Atoms,
        timestep: float,
        temperature_K: float,
        model: AtomisticModel,
        device: str | torch.device = "auto",
        time_constant: float = 100.0 * ase.units.fs,
        fixcm: bool = True,
        rescale_energy: bool = False,
        random_rotation: bool = False,
        **kwargs,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        stepper = FlashMDStepper(model, device)
        dtype = stepper.dtype
        return cls(atoms, timestep, temperature_K, stepper, device, dtype, time_constant, fixcm, rescale_energy, random_rotation, **kwargs)

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
            self.atoms.set_velocities(
                self.atoms.get_velocities()
                - self.atoms.get_momenta().sum(axis=0) / self.atoms.get_masses().sum()
            )
