from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .fpi import _anderson_update
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)


class SymplecticModule(torch.nn.Module):
    """TorchScript-compatible module wrapping the symplectic FlashMD step.

    Runs the FlashMD model for an initial guess, then refines via an inlined
    Anderson-accelerated midpoint fixed-point iteration using the symplectic
    correction model. Neighbor lists from the input system are reused at each
    midpoint estimate (valid as long as positional changes are smaller than the
    neighbor list skin).

    Output metadata (Labels) is taken from the inner model outputs rather than
    constructed fresh, which is required for TorchScript serialization.
    """

    # NL options stored as primitive lists so TorchScript can serialize them
    _nl_cutoffs: List[float]
    _nl_full_lists: List[bool]
    _nl_stricts: List[bool]
    tol: float
    max_iter: int
    m: int
    beta: float
    lambda_reg: float
    timestep: torch.Tensor  # scalar tensor in fs; LAMMPS reads it via .toTensor()
    verbose: bool
    _printed_config: bool

    def __init__(
        self,
        flashmd_model: AtomisticModel,
        symplectic_model: AtomisticModel,
        config: Dict[str, float],
        symplectic_nl_options: List[NeighborListOptions],
        verbose: bool = False,
    ):
        super().__init__()
        # Store inner modules directly - AtomisticModel wrappers can't be
        # nested inside another module for TorchScript serialization
        self.flashmd_module = flashmd_model.module
        self.symplectic_module = symplectic_model.module
        # NeighborListOptions aren't serializable; store as primitive lists
        self._nl_cutoffs = [nl.cutoff for nl in symplectic_nl_options]
        self._nl_full_lists = [nl.full_list for nl in symplectic_nl_options]
        self._nl_stricts = [nl.strict for nl in symplectic_nl_options]

        self.tol = float(config.get("tol", 1e-5))
        self.max_iter = int(config.get("max_iter", 50))
        self.m = int(config.get("m", 5))
        self.beta = float(config.get("beta", 0.9))
        self.lambda_reg = float(config.get("lambda_reg", 1e-4))
        self.verbose = verbose
        self._printed_config = False
        ts = flashmd_model.module.timestep
        self.register_buffer(
            "timestep",
            ts.clone() if isinstance(ts, torch.Tensor) else torch.tensor(float(ts)),
        )

    def _make_midpoint_system(
        self,
        q_bar: torch.Tensor,
        p_bar: torch.Tensor,
        template: System,
        template_out: Dict[str, TensorMap],
    ) -> System:
        """Build a midpoint System reusing metadata and NLs from *template*."""
        system = System(
            positions=q_bar,
            types=template.types,
            cell=template.cell,
            pbc=template.pbc,
        )
        # Reuse sample/component/property Labels from the existing output blocks
        mom_block = template_out["momenta"].block()
        mass_block = template.get_data("masses").block()
        system.add_data(
            "momenta",
            TensorMap(
                keys=template_out["momenta"].keys,
                blocks=[TensorBlock(
                    values=p_bar if p_bar.dim() == 3 else p_bar.unsqueeze(-1),
                    samples=mom_block.samples,
                    components=mom_block.components,
                    properties=mom_block.properties,
                )],
            ),
        )
        system.add_data(
            "masses",
            TensorMap(
                keys=template.get_data("masses").keys,
                blocks=[TensorBlock(
                    values=mass_block.values,
                    samples=mass_block.samples,
                    components=mass_block.components,
                    properties=mass_block.properties,
                )],
            ),
        )
        for i in range(len(self._nl_cutoffs)):
            nl_options = NeighborListOptions(
                cutoff=self._nl_cutoffs[i],
                full_list=self._nl_full_lists[i],
                strict=self._nl_stricts[i],
            )
            nl_block = template.get_neighbor_list(nl_options)
            system.add_neighbor_list(nl_options, nl_block)
        return system

    def _midpoint_map(
        self,
        system: System,
        template_out: Dict[str, TensorMap],
        x_init: torch.Tensor,
        x_bar: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """One evaluation of the midpoint fixed-point mapping."""
        q_bar = x_bar[: n * 3].view(n, 3)
        p_bar = x_bar[n * 3 :].view(n, 3, 1)

        midpoint = self._make_midpoint_system(q_bar, p_bar, system, template_out)
        sym_outputs: Dict[str, ModelOutput] = {
            "positions": ModelOutput(per_atom=True),
            "momenta": ModelOutput(per_atom=True),
        }
        outputs = self.symplectic_module([midpoint], sym_outputs, None)
        delta_q = outputs["positions"].block().values.squeeze(-1)
        delta_p = outputs["momenta"].block().values

        delta = torch.cat([delta_q.flatten(), delta_p.flatten()])
        return x_init + 0.5 * delta

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        flashmd_outputs: Dict[str, ModelOutput] = {
            "positions": ModelOutput(per_atom=True),
            "momenta": ModelOutput(per_atom=True),
        }
        # Run FlashMD for initial guess - also gives us the output metadata
        guess_out = self.flashmd_module(systems, flashmd_outputs, None)

        # Process one system at a time (LAMMPS sends one)
        result_pos_blocks: List[TensorBlock] = []
        result_mom_blocks: List[TensorBlock] = []

        for i in range(len(systems)):
            system = systems[i]
            n = system.positions.shape[0]

            # Extract initial guess for this system
            # (single-system case: block index 0; multi-system: use per-system blocks)
            pos_block = guess_out["positions"].block()
            mom_block = guess_out["momenta"].block()
            q_prime = pos_block.values.squeeze(-1)
            p_prime = mom_block.values.squeeze(-1)

            x_init = torch.cat(
                [system.positions.flatten(), system.get_data("momenta").block().values.flatten()]
            )
            x_bar = 0.5 * (x_init + torch.cat([q_prime.flatten(), p_prime.flatten()]))

            # NOTE: the loop is inlined here rather than calling anderson_solver()
            # because TorchScript does not support arbitrary Callable arguments -
            # the fixed-point mapping (self._midpoint_map) can only be invoked
            # as a method, not passed as a function. _anderson_update is reused
            # for the actual update step to avoid duplicating the linear algebra.
            delta_xs: List[torch.Tensor] = []
            delta_gs: List[torch.Tensor] = []

            fx = self._midpoint_map(system, guess_out, x_init, x_bar, n)
            g = fx - x_bar
            x_prev = x_bar.clone()
            g_prev = g.clone()

            n_iters: int = 0
            for k in range(self.max_iter):
                if torch.norm(g) < self.tol:
                    break

                if k > 0:
                    delta_xs.append(x_bar - x_prev)
                    delta_gs.append(g - g_prev)
                    if len(delta_xs) > self.m:
                        delta_xs = delta_xs[1:]
                        delta_gs = delta_gs[1:]

                x_prev = x_bar
                g_prev = g

                x_bar = _anderson_update(
                    x_bar, g, delta_xs, delta_gs, self.beta, self.lambda_reg
                )

                fx = self._midpoint_map(system, guess_out, x_init, x_bar, n)
                g = fx - x_bar
                n_iters = k + 1

            if self.verbose:
                if not self._printed_config:
                    print(
                        "SymplecticModule config: tol=", self.tol,
                        "max_iter=", self.max_iter,
                        "m=", self.m,
                        "beta=", self.beta,
                        "lambda_reg=", self.lambda_reg,
                    )
                    self._printed_config = True
                print("FPI:", n_iters, "iters, |g| =", torch.norm(g).item())

            # Recover endpoint and reuse output metadata from inner model
            x_star = 2 * x_bar - x_init
            q_star = x_star[: n * 3].view(n, 3)
            p_star = x_star[n * 3 :].view(n, 3, 1)

            result_pos_blocks.append(TensorBlock(
                values=q_star.unsqueeze(-1),
                samples=pos_block.samples,
                components=pos_block.components,
                properties=pos_block.properties,
            ))
            result_mom_blocks.append(TensorBlock(
                values=p_star,
                samples=mom_block.samples,
                components=mom_block.components,
                properties=mom_block.properties,
            ))

        return {
            "positions": TensorMap(keys=guess_out["positions"].keys, blocks=result_pos_blocks),
            "momenta": TensorMap(keys=guess_out["momenta"].keys, blocks=result_mom_blocks),
        }


def export_symplectic_model(
    flashmd_model: AtomisticModel,
    symplectic_model: AtomisticModel,
    config: Optional[Dict[str, float]] = None,
    metadata: Optional[ModelMetadata] = None,
    verbose: bool = False,
) -> AtomisticModel:
    """Wrap a FlashMD model and its symplectic correction into an exportable AtomisticModel.

    The returned model has the same interface as a regular FlashMD model and can be
    saved with ``.save()`` for use with LAMMPS (via ``fix metatomic``).

    Args:
        flashmd_model: Pre-trained FlashMD AtomisticModel (initial guess).
        symplectic_model: Symplectic correction AtomisticModel.
        config: Anderson solver hyper-parameters (tol, max_iter, m, beta, lambda_reg).
        metadata: Optional ModelMetadata for the exported model.
        verbose: If True, print FPI iteration count and residual norm each step.

    Returns:
        An AtomisticModel ready for ``.save()``.
    """
    if config is None:
        config = {}
    if metadata is None:
        metadata = ModelMetadata()

    symplectic_nl_options = symplectic_model.requested_neighbor_lists()
    module = SymplecticModule(flashmd_model, symplectic_model, config, symplectic_nl_options, verbose)
    module.eval()

    base_caps = flashmd_model.capabilities()
    capabilities = ModelCapabilities(
        outputs={
            "positions": ModelOutput(per_atom=True),
            "momenta": ModelOutput(per_atom=True),
        },
        atomic_types=base_caps.atomic_types,
        interaction_range=max(
            base_caps.interaction_range,
            symplectic_model.capabilities().interaction_range,
        ),
        length_unit=base_caps.length_unit,
        supported_devices=base_caps.supported_devices,
        dtype=base_caps.dtype,
    )

    return AtomisticModel(module, metadata, capabilities)
