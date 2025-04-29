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


def get_energy_error(file_path, model_path, skipmd_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MetatensorCalculator(model_path, extensions_directory="extensions_pet_mad/", device=device)
    skipmd_model = load_atomistic_model(skipmd_model_path)
    skipmd_model = skipmd_model.to(device)

    structures = ase.io.read(file_path, index="::10")

    evaluation_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs=skipmd_model.capabilities().outputs,
    )

    delta_q_key = [k for k in skipmd_model.capabilities().outputs.keys() if "mtt::delta_" in k][0]
    p_key = [k for k in skipmd_model.capabilities().outputs.keys() if "mtt::p_" in k][0]

    n_steps = int(delta_q_key.split("_")[1].split("_")[0])
    energies_md = []
    energies_skipmd = []

    for structure in tqdm.tqdm(structures):
        atoms = copy.deepcopy(structure)
        masses = torch.tensor(atoms.get_masses(), device=device)
        system = _atoms_to_system(atoms, device)
        system = get_system_with_neighbor_lists(system, skipmd_model.requested_neighbor_lists())

        atoms.calc = calculator
        # actually do not run MD
        # integrator = VelocityVerlet(atoms, 0.5 * ase.units.fs)
        # integrator.run(n_steps)
        energies_md.append(atoms.get_total_energy()/len(atoms))

        results = skipmd_model([system], evaluation_options, check_consistency=False)
        system.positions[:] += results[delta_q_key].block().values.squeeze(-1) / torch.sqrt(masses[:, None])
        system.get_data("momenta").block().values[:] = results[p_key].block().values * torch.sqrt(masses[:, None, None])

        atoms_skipmd = _system_to_atoms(system)
        atoms_skipmd.calc = calculator
        energies_skipmd.append(atoms_skipmd.get_total_energy()/len(atoms))

    rmse = np.sqrt(np.mean((np.array(energies_md) - np.array(energies_skipmd))**2))

    # plot scatter plot
    plt.scatter(energies_md, energies_skipmd, s=1)
    plt.xlabel("MD energy")
    plt.ylabel("SkipMD energy")
    plt.title(f"SkipMD vs MD energy (RMSE: {rmse:.8f})")
    plt.plot([min(energies_md), max(energies_md)], [min(energies_md), max(energies_md)], color='red', linestyle='--')
    plt.savefig("energy_error.pdf")

    # also plot the absolute error as a histogram


def _atoms_to_system(atoms, device):
    system = systems_to_torch(atoms, dtype=torch.float32, device=device)
    system.add_data(
        "momenta",
        metatensor.torch.TensorMap(
            keys=metatensor.torch.Labels.single().to(device),
            blocks = [
                metatensor.torch.TensorBlock(
                    values=torch.tensor(atoms.get_momenta(), dtype=torch.float32, device=device).unsqueeze(-1),
                    samples=metatensor.torch.Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, j] for j in range(len(atoms))], device=device),
                    ),
                    components=[metatensor.torch.Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=device))],
                    properties=metatensor.torch.Labels.single().to(device),
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
