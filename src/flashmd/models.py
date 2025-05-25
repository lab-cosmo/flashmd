from huggingface_hub import hf_hub_download
from metatensor.torch.atomistic import MetatensorAtomisticModel, load_atomistic_model


def get_universal_model(time_step: int = 16) -> MetatensorAtomisticModel:

    if (time_step not in [1, 4, 16, 64]):
        raise ValueError("Univeral FlashMD models are only available for"
                         " time steps of 1fs, 4fs, 16fs, 64fs.")

    model_path = hf_hub_download(
        repo_id="lab-cosmo/flashmd",
        filename=f"flashmd_{time_step}fs.pt",
        cache_dir=None,
        revision="main"
    )
    return load_atomistic_model(model_path)
