import functools

import pytest

import flashmd.models


@pytest.fixture(scope="session", autouse=True)
def _cache_hf_downloads():
    """Memoise Hugging Face downloads for the whole test session.

    Several tests call ``get_pretrained``, which re-downloads the shared MLIP
    checkpoint on every call. That burst of requests gets rate-limited (and
    rejected) by the Hub in CI. Caching ``hf_hub_download`` by its arguments
    means each file is fetched at most once per session.
    """
    original = flashmd.models.hf_hub_download
    flashmd.models.hf_hub_download = functools.cache(original)
    yield
    flashmd.models.hf_hub_download = original
