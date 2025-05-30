from huggingface_hub import hf_hub_download
from metatomic.torch import AtomisticModel, load_atomistic_model


def get_universal_model(time_step: int = 16) -> AtomisticModel:
    if time_step not in [1, 4, 8, 16, 32, 64]:
        raise ValueError(
            "Universal FlashMD models are only available for"
            " time steps of 1, 4, 8, 16, 32, 64 fs."
        )

    model_path = hf_hub_download(
        repo_id="lab-cosmo/flashmd",
        filename=f"flashmd_{time_step}fs.pt",
        cache_dir=None,
        revision="main",
    )
    return load_atomistic_model(model_path)
