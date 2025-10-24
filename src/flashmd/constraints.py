from typing import Dict, List

import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import System


@torch.jit.script
def enforce_physical_constraints(
    systems: List[System],
    predictions: Dict[str, TensorMap],
    timestep: float,
) -> Dict[str, TensorMap]:
    """
    Enforces physical constraints in the predictions of a FlashMD model, namely
    conservation of momentum of the center of mass and uniform linear motion of the
    center of mass.
    """

    new_predictions: Dict[str, TensorMap] = {}

    for key, prediction_tmap in predictions.items():
        if key == "momenta":
            # conservation of momentum of the center of mass
            system_sizes = [len(s) for s in systems]
            masses = [s.get_data("masses").block().values for s in systems]
            total_masses = [m.sum() for m in masses]
            momenta_before = [s.get_data("momenta").block().values for s in systems]
            momenta_now = torch.split(prediction_tmap.block().values, system_sizes)
            velocities_now = [p / m[:, None] for p, m in zip(momenta_now, masses)]
            velocities_com_before = [
                torch.sum(p, dim=0) / M for p, M in zip(momenta_before, total_masses)
            ]
            velocities_com_now = [
                torch.sum(p, dim=0) / M for p, M in zip(momenta_now, total_masses)
            ]
            velocities_now = [
                v - v_com_now_i + v_com_before_i
                for v, v_com_before_i, v_com_now_i in zip(
                    velocities_now, velocities_com_before, velocities_com_now
                )
            ]
            momenta_now = [v * m[:, None] for v, m in zip(velocities_now, masses)]
            new_predictions[key] = TensorMap(
                prediction_tmap.keys,
                [
                    TensorBlock(
                        values=torch.concatenate(momenta_now),
                        samples=prediction_tmap.block().samples,
                        components=prediction_tmap.block().components,
                        properties=prediction_tmap.block().properties,
                    )
                ],
            )
        elif key == "positions":
            # uniform linear motion of the center of mass
            system_sizes = [len(s) for s in systems]
            masses = [s.get_data("masses").block().values for s in systems]
            total_masses = [m.sum() for m in masses]
            positions_before = [s.positions.unsqueeze(-1) for s in systems]
            momenta = [s.get_data("momenta").block().values for s in systems]
            positions_now = torch.split(prediction_tmap.block().values, system_sizes)
            velocities_com = [
                torch.sum(p, dim=0) / M for p, M in zip(momenta, total_masses)
            ]
            positions_com_before = [
                torch.sum(q * m[:, None], dim=0) / M
                for q, m, M in zip(positions_before, masses, total_masses)
            ]
            positions_com_now = [
                torch.sum(q * m[:, None], dim=0) / M
                for q, m, M in zip(positions_now, masses, total_masses)
            ]
            positions_now = [
                q - q_com_now_i + q_com_before_i + v_com_i * timestep
                for q, q_com_now_i, q_com_before_i, v_com_i in zip(
                    positions_now,
                    positions_com_now,
                    positions_com_before,
                    velocities_com,
                )
            ]
            new_predictions[key] = TensorMap(
                prediction_tmap.keys,
                [
                    TensorBlock(
                        values=torch.concatenate(positions_now),
                        samples=prediction_tmap.block().samples,
                        components=prediction_tmap.block().components,
                        properties=prediction_tmap.block().properties,
                    )
                ],
            )
        else:
            new_predictions[key] = prediction_tmap

    return new_predictions
