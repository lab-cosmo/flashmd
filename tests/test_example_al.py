"""Test that the Al example runs end-to-end on CPU with a tiny model."""

import os
import tempfile
from pathlib import Path

import yaml


EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "al"

_MINIMAL_MODEL_HYPERS = {
    "d_pet": 1,
    "d_head": 1,
    "d_node": 1,
    "d_feedforward": 1,
    "num_heads": 1,
    "num_attention_layers": 1,
    "num_gnn_layers": 1,
}


def _modify_al_py(code: str) -> str:
    # Swap out the MLIP calculator for EMT (no GPU, no download required).
    code = code.replace(
        "from upet.calculator import UPETCalculator",
        "from ase.calculators.emt import EMT",
    )
    code = code.replace(
        'atoms.calc = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cuda")',
        "atoms.calc = EMT()",
    )

    # Reduce the number of MD steps so the test finishes quickly.
    code = code.replace(
        "Langevin(atoms, 2 * units.fs, temperature_K=400, friction=gamma).run(1000)",
        "Langevin(atoms, 2 * units.fs, temperature_K=400, friction=gamma).run(5)",
    )
    code = code.replace("trange(1000)", "trange(30)")
    code = code.replace("num_decorrelation_frames = 10", "num_decorrelation_frames = 2")

    # Reduce i-PI simulation steps.
    code = code.replace("simulation.run(100)", "simulation.run(5)")
    code = code.replace(
        "symplectic_simulation.run(100)", "symplectic_simulation.run(5)"
    )

    return code


def _modify_training_yaml(path: Path, architecture_name: str) -> str:
    with open(path) as f:
        hypers = yaml.safe_load(f)

    hypers["architecture"]["model"] = _MINIMAL_MODEL_HYPERS.copy()
    hypers["architecture"]["training"]["num_epochs"] = 2
    hypers["architecture"]["training"]["batch_size"] = 2

    return yaml.dump(hypers)


def _modify_simulation_xml(xml: str) -> str:
    # Replace the metatomic force field with a dummy PES so no model file is
    # needed (FlashMD replaces the motion step entirely anyway).
    xml = xml.replace("<pes>metatomic</pes>", "<pes>dummy</pes>")
    xml = xml.replace(
        "<parameters>{model:./pet-mad-s-v1.5.0.pt, template:./al.xyz, device:cuda}</parameters>",
        "<parameters>{}</parameters>",
    )
    xml = xml.replace("<total_steps>100</total_steps>", "<total_steps>5</total_steps>")
    return xml


def test_example_al():
    code = _modify_al_py((EXAMPLE_DIR / "al.py").read_text())
    flashmd_yaml = _modify_training_yaml(
        EXAMPLE_DIR / "options-flashmd.yaml", "experimental.flashmd"
    )
    symplectic_yaml = _modify_training_yaml(
        EXAMPLE_DIR / "options-symplectic-flashmd.yaml",
        "experimental.flashmd_symplectic",
    )
    simulation_xml = _modify_simulation_xml(
        (EXAMPLE_DIR / "simulation-template.xml").read_text()
    )

    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / "options-flashmd.yaml").write_text(flashmd_yaml)
        (Path(tmp_dir) / "options-symplectic-flashmd.yaml").write_text(symplectic_yaml)
        (Path(tmp_dir) / "simulation-template.xml").write_text(simulation_xml)

        try:
            os.chdir(tmp_dir)
            exec(code, {})  # noqa: S102
        finally:
            os.chdir(original_dir)
