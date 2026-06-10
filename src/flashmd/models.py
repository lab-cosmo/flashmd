import os
import shutil
import subprocess
import time

from huggingface_hub import hf_hub_download
from metatomic.torch import load_atomistic_model


AVAILABLE_MLIPS = ["pet-omatpes", "pet-omatpes-v2"]
AVAILABLE_TIME_STEPS = {
    "pet-omatpes": [1, 2, 4, 8, 16, 32, 64, 128],
    "pet-omatpes-v2": [1, 2, 4, 8, 16, 32, 64, 128],
}
AVAILABLE_SYMPLECTIC_TIME_STEPS = {
    "pet-omatpes": [2, 16],
}


def get_pretrained(
    mlip: str = "pet-omatpes-v2", time_step: int = 16, symplectic: bool = False
):
    if mlip not in AVAILABLE_MLIPS:
        raise ValueError(
            f"MLIP '{mlip}' is not available. "
            f"Available MLIPs are: {', '.join(AVAILABLE_MLIPS)}."
        )

    if time_step not in AVAILABLE_TIME_STEPS[mlip]:
        raise ValueError(
            f"Pre-trained FlashMD models based on the {mlip} MLIP are only available "
            f"for time steps of {', '.join(map(str, AVAILABLE_TIME_STEPS[mlip]))} fs."
        )

    if symplectic:
        if mlip not in AVAILABLE_SYMPLECTIC_TIME_STEPS:
            raise ValueError(
                f"No symplectic FlashMD model is available for the {mlip} MLIP."
            )
        if time_step not in AVAILABLE_SYMPLECTIC_TIME_STEPS[mlip]:
            raise ValueError(
                f"Symplectic FlashMD models based on the {mlip} MLIP are only available "
                f"for time steps of {', '.join(map(str, AVAILABLE_SYMPLECTIC_TIME_STEPS[mlip]))} fs."
            )
    # Get checkpoints corresponding to the selected MLIP and FlashMD models
    mlip_path = hf_hub_download(
        repo_id="lab-cosmo/flashmd",
        filename=f"mlip_{mlip}.ckpt",
        cache_dir=None,
        revision="main",
    )
    flashmd_path = hf_hub_download(
        repo_id="lab-cosmo/flashmd",
        filename=f"flashmd_{mlip}_{time_step}fs.ckpt",
        cache_dir=None,
        revision="main",
    )

    # Now we need to export both using metatrain. However, we don't want to do it if
    # HuggingFace hasn't downloaded a new version of the files, so we only re-export
    # if the files above have changed in the last 10 seconds.
    reexport = False
    exported_mlip_path = mlip_path.replace(".ckpt", ".pt")
    exported_flashmd_path = flashmd_path.replace(".ckpt", ".pt")
    if not os.path.exists(exported_mlip_path) or not os.path.exists(
        exported_flashmd_path
    ):
        reexport = True
    mlip_mtime = os.path.getmtime(mlip_path)
    flashmd_mtime = os.path.getmtime(flashmd_path)
    if (time.time() - mlip_mtime < 10) or (time.time() - flashmd_mtime < 10):
        reexport = True
    if reexport:
        subprocess.run(
            ["mtt", "export", mlip_path, "-o", exported_mlip_path], capture_output=True
        )
        subprocess.run(
            ["mtt", "export", flashmd_path, "-o", exported_flashmd_path],
            capture_output=True,
        )

    # Load as AtomisticModel instances.
    # If it doesn't work, try to re-export once  and load again (this will, among
    # others, catch upgrades in metatomic that break compatibility)
    try:
        mlip_model = load_atomistic_model(exported_mlip_path)
        flashmd_model = load_atomistic_model(exported_flashmd_path)
    except Exception:
        subprocess.run(
            ["mtt", "export", mlip_path, "-o", exported_mlip_path], capture_output=True
        )
        subprocess.run(
            ["mtt", "export", flashmd_path, "-o", exported_flashmd_path],
            capture_output=True,
        )
        mlip_model = load_atomistic_model(exported_mlip_path)
        flashmd_model = load_atomistic_model(exported_flashmd_path)

    if not symplectic:
        return mlip_model, flashmd_model

    symplectic_path = hf_hub_download(
        repo_id="lab-cosmo/flashmd",
        filename=f"flashmd-symplectic_{mlip}_{time_step}fs.ckpt",
        cache_dir=None,
        revision="main",
    )
    exported_symplectic_path = symplectic_path.replace(".ckpt", ".pt")
    symplectic_reexport = False
    if not os.path.exists(exported_symplectic_path):
        symplectic_reexport = True
    if time.time() - os.path.getmtime(symplectic_path) < 10:
        symplectic_reexport = True
    if symplectic_reexport:
        result = subprocess.run(
            ["mtt", "export", symplectic_path, "-o", exported_symplectic_path],
            capture_output=True,
        )
        if result.returncode != 0:
            try:
                import metatrain.experimental.flashmd_symplectic  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Using symplectic FlashMD models requires a version of metatrain "
                    "that includes the experimental.flashmd_symplectic architecture. "
                    "Please install it from the main branch:\n"
                    "    pip install 'metatrain @ git+https://github.com/metatensor/metatrain.git'"
                )
            raise RuntimeError(result.stderr.decode())

    try:
        symplectic_model = load_atomistic_model(exported_symplectic_path)
    except Exception:
        print(f"{symplectic_path=}")
        print(f"{exported_symplectic_path=}")
        result = subprocess.run(
            ["mtt", "export", symplectic_path, "-o", exported_symplectic_path],
            capture_output=True,
        )
        if result.returncode != 0:
            try:
                import metatrain.experimental.flashmd_symplectic  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Using symplectic FlashMD models requires a version of metatrain "
                    "that includes the experimental.flashmd_symplectic architecture. "
                    "Please install it from the main branch:\n"
                    "    pip install 'metatrain @ git+https://github.com/metatensor/metatrain.git'"
                )
            raise RuntimeError(result.stderr.decode())
        print(f"{result.stdout=}")
        symplectic_model = load_atomistic_model(exported_symplectic_path)

    return mlip_model, flashmd_model, symplectic_model


def save_checkpoint(mlip: str = "pet-omatpes-v2", time_step: int = 16):
    if time_step not in AVAILABLE_TIME_STEPS[mlip]:
        raise ValueError(
            f"Pre-trained FlashMD models based on the {mlip} MLIP are only available "
            f"for time steps of {', '.join(map(str, AVAILABLE_TIME_STEPS[mlip]))} fs."
        )

    checkpoint_path = hf_hub_download(
        repo_id="lab-cosmo/flashmd",
        filename=f"flashmd_{mlip}_{time_step}fs.ckpt",
        cache_dir=None,
        revision="main",
    )

    # Copy it to the current directory
    shutil.copyfile(checkpoint_path, f"flashmd_{mlip}_{time_step}fs.ckpt")
