from abc import ABC, abstractmethod

from metatomic.torch import System


class AtomisticStepper(ABC):
    @abstractmethod
    def get_timestep(self) -> float:
        """Get the time step of the stepper in femtoseconds.

        Returns:
            float: The time step in femtoseconds.
        """

    @abstractmethod
    def step(self, system: System) -> System:  # type: ignore
        """Perform a single MD step on the given system.

        Args:
            system (System): The input system containing positions, momenta, etc.

        Returns:
            System: The updated system after one MD step.
        """
