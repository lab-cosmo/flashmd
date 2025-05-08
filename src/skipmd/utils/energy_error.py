import ase.io
from metatensor.torch.atomistic import load_atomistic_model, systems_to_torch, ModelEvaluationOptions
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
import torch
from ase.md import VelocityVerlet
import copy
import metatensor.torch
import matplotlib.pyplot as plt
import tqdm
import ase.units
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
import numpy as np
from ..stepper import SkipMDStepper
from metatensor.torch import Labels, TensorBlock, TensorMap


def get_energy_error(file_path, model_path, skipmd_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MetatensorCalculator(model_path, extensions_directory="extensions_pet_mad/", device=device)
    skipmd_model = load_atomistic_model(skipmd_model_path)

    delta_q_key = [k for k in skipmd_model.capabilities().outputs.keys() if "mtt::delta_" in k][0]
    n_steps = int(delta_q_key.split("_")[1].split("_")[0])

    stepper = SkipMDStepper([skipmd_model], n_steps, torch.device(device))

    structures = ase.io.read(file_path, index="::10")

    energies_md = []
    energies_skipmd = []

    for structure in tqdm.tqdm(structures):
        atoms = copy.deepcopy(structure)
        atoms.calc = calculator
        energies_md.append(atoms.get_total_energy()/len(atoms))
        system = _atoms_to_system(atoms, device)

        new_system = stepper.step(system)

        atoms_skipmd = _system_to_atoms(new_system)
        atoms_skipmd.calc = calculator
        energies_skipmd.append(atoms_skipmd.get_total_energy()/len(atoms))

    rmse = np.sqrt(np.mean((np.array(energies_md) - np.array(energies_skipmd))**2))

    # plot scatter plot
    plt.scatter(energies_md, energies_skipmd, s=1)
    plt.xlabel("MD energy")
    plt.ylabel("FlashMD energy")
    plt.title(f"FlashMD vs MD energy (RMSE: {rmse:.8f})")
    plt.plot([min(energies_md), max(energies_md)], [min(energies_md), max(energies_md)], color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig("energy_error.pdf")

    # also plot the absolute error as a histogram


def _atoms_to_system(atoms, device):
    system = systems_to_torch(atoms, dtype=torch.float32, device=device)
    system.add_data(
        "momenta",
        TensorMap(
            keys=Labels.single().to(device),
            blocks = [
                TensorBlock(
                    values=torch.tensor(atoms.get_momenta(), dtype=torch.float32, device=device).unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, j] for j in range(len(atoms))], device=device),
                    ),
                    components=[Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=device))],
                    properties=Labels.single().to(device),
                )
            ],
        )
    )
    system.add_data(
        "masses",
        TensorMap(
            keys=Labels.single().to(device),
            blocks = [
                TensorBlock(
                    values=torch.tensor(atoms.get_masses(), dtype=torch.float32, device=device).unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, j] for j in range(len(atoms))], device=device),
                    ),
                    components=[],
                    properties=Labels.single().to(device),
                )
            ],
        )
    )
    return system

def _system_to_atoms(system):
    atomic_numbers = system.types.cpu().numpy()
    positions = system.positions.cpu().numpy()
    cell = system.cell.cpu().numpy()
    pbc = system.pbc.cpu().numpy()
    momenta = system.get_data("momenta").block().values.squeeze(-1).cpu().numpy()
    atoms = ase.Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
        momenta=momenta,
    )
    return atoms
