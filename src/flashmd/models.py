from huggingface_hub import hf_hub_download
from metatomic.torch import AtomisticModel, load_atomistic_model
import subprocess
import os
import time


AVAILABLE_MLIPS = ["pet-omatpes"]
AVAILABLE_TIME_STEPS = {
    "pet-omatpes": [1, 2, 4, 8, 16, 32, 64, 128]
}


def get_pretrained(mlip: str = "pet-omatpes", time_step: int = 16) -> AtomisticModel:
    if mlip not in AVAILABLE_MLIPS:
        raise ValueError(
            f"MLIP '{mlip}' is not available. "
            f"Available MLIPs are: {', '.join(AVAILABLE_MLIPS)}."
        )

    if time_step not in AVAILABLE_TIME_STEPS[mlip]:
        raise ValueError(
            f"Pretrained FlashMD models based on the {mlip} MLIP are only available "
            f"for time steps of {', '.join(map(str, AVAILABLE_TIME_STEPS[mlip]))} fs."
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
    if not os.path.exists(exported_mlip_path) or not os.path.exists(exported_flashmd_path):
        reexport = True
    mlip_mtime = os.path.getmtime(mlip_path)
    flashmd_mtime = os.path.getmtime(flashmd_path)
    if (time.time() - mlip_mtime < 10) or (time.time() - flashmd_mtime < 10):
        reexport = True
    if reexport:
        subprocess.run(["mtt", "export", mlip_path, "-o", exported_mlip_path], capture_output=True)
        subprocess.run(["mtt", "export", flashmd_path, "-o", exported_flashmd_path], capture_output=True)

    # Load as AtomisticModel instances
    mlip_model = load_atomistic_model(exported_mlip_path)
    flashmd_model = load_atomistic_model(exported_flashmd_path)

    return mlip_model, flashmd_model
