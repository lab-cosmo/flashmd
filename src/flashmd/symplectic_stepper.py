from functools import partial

import torch
import vesin.metatomic
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import AtomisticModel, ModelEvaluationOptions, ModelOutput, System

from flashmd.fpi import anderson_solver


class SymplecticStepper:
    """Fixed-point iteration based symplectic integrator.

    Uses a FlashMDStepper for the initial guess, then refines via Anderson
    acceleration on the midpoint fixed-point problem.

    The fixed-point problem is: find the midpoint ``x_bar`` in phase space such
    that ``x_bar = x_init + 0.5 * delta(x_bar)``, where ``delta`` is the output
    of the symplectic correction model evaluated at ``x_bar``. The converged
    endpoint is then ``x_star = 2 * x_bar - x_init``.

    Args:
        flashmd_stepper: Pre-built FlashMDStepper used to generate the initial guess.
        symplectic_model: AtomisticModel evaluated at the midpoint; must output
            ``positions`` and ``momenta`` keys representing the full-step deltas.
        config: Optional dict with keys:
            - ``tol`` (float, default 1e-5): convergence tolerance for Anderson
            - ``max_iter`` (int, default 50): maximum Anderson iterations
            - ``m`` (int, default 5): Anderson history size
            - ``beta`` (float, default 1.0): Anderson mixing parameter
            - ``lambda_reg`` (float, default 1e-4): Anderson regularisation
    """

    def __init__(
        self,
        flashmd_stepper,
        symplectic_model: AtomisticModel,
        config: dict | None = None,
    ):
        if config is None:
            config = {}
        self.flashmd_stepper = flashmd_stepper
        self.device = flashmd_stepper.device
        self.symplectic_model = symplectic_model.to(self.device)

        flashmd_timestep = float(flashmd_stepper.model.module.timestep)
        symplectic_timestep = float(symplectic_model.module.timestep)
        if symplectic_timestep != flashmd_timestep:
            raise ValueError(
                f"Mismatch between FlashMD model timestep ({flashmd_timestep} fs) "
                f"and symplectic model timestep ({symplectic_timestep} fs)."
            )

        self.tol = config.get("tol", 1e-5)
        self.max_iter = config.get("max_iter", 50)
        self.m = config.get("m", 5)
        self.beta = config.get("beta", 1.0)
        self.lambda_reg = config.get("lambda_reg", 1e-4)

        self.evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={
                "positions": ModelOutput(per_atom=True),
                "momenta": ModelOutput(per_atom=True),
            },
        )
        self.neighbor_list_calculators = vesin.metatomic.neighbor_lists_for_model(
            "angstrom", self.symplectic_model
        )
        # vesin's CUDA brute_force algorithm is broken for triclinic cells
        for nl in self.neighbor_list_calculators:
            nl._nl.algorithm = "cell_list"

    def _midpoint_step(
        self, system: System, x_init: torch.Tensor, x_bar: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the fixed-point mapping at midpoint estimate ``x_bar``.

        Returns the new midpoint estimate ``x_init + 0.5 * delta(x_bar)``.
        """
        n = system.positions.shape[0]
        masses = system.get_data("masses").block().values

        q_bar = x_bar[: n * 3].view(n, 3)
        p_bar = x_bar[n * 3 :].view(n, 3, 1)

        midpoint = _make_system(q_bar, p_bar, masses, system)
        for calculator in self.neighbor_list_calculators:
            calculator.add_neighbor_list(midpoint)

        outputs = self.symplectic_model(
            [midpoint], self.evaluation_options, check_consistency=False
        )
        delta_q = outputs["positions"].block().values.squeeze(-1)
        delta_p = outputs["momenta"].block().values

        delta = torch.cat([delta_q.flatten(), delta_p.flatten()])
        return x_init + 0.5 * delta

    def step(self, system: System) -> System:
        n = system.positions.shape[0]
        masses = system.get_data("masses").block().values

        # flatten initial state to a phase-space vector
        x_init = torch.cat(
            [
                system.positions.flatten(),
                system.get_data("momenta").block().values.flatten(),
            ]
        )

        # FlashMD initial guess → initial midpoint estimate
        initial_guess = self.flashmd_stepper.step(system)
        x_prime = torch.cat(
            [
                initial_guess.positions.flatten(),
                initial_guess.get_data("momenta").block().values.flatten(),
            ]
        )
        x_bar_init = 0.5 * (x_init + x_prime)

        # Anderson acceleration on the midpoint fixed-point problem
        f = partial(self._midpoint_step, system, x_init)
        x_bar_star = anderson_solver(
            f,
            x_bar_init,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            beta=self.beta,
            lambda_reg=self.lambda_reg,
        )

        # recover endpoint from converged midpoint
        x_star = 2 * x_bar_star - x_init
        q_star = x_star[: n * 3].view(n, 3)
        p_star = x_star[n * 3 :].view(n, 3, 1)

        return _make_system(q_star, p_star, masses, system)


def _make_system(positions, momenta, masses, template: System) -> System:
    """Build a System from updated positions/momenta, copying metadata from template."""
    device = template.positions.device
    n = positions.shape[0]
    atom_samples = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, j] for j in range(n)], device=device),
    )
    xyz = [Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=device))]

    system = System(
        positions=positions,
        types=template.types,
        cell=template.cell,
        pbc=template.pbc,
    )
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks=[
                TensorBlock(
                    values=momenta if momenta.dim() == 3 else momenta.unsqueeze(-1),
                    samples=atom_samples,
                    components=xyz,
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
                        values=torch.tensor([[0, j] for j in range(n)], device=device),
                    ),
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        ),
    )
    return system
