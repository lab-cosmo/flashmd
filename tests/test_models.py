import pytest

from flashmd.models import AVAILABLE_MLIPS, AVAILABLE_TIME_STEPS


def test_available_mlips():
    """Test that AVAILABLE_MLIPS is defined and contains expected values."""
    assert isinstance(AVAILABLE_MLIPS, list)
    assert len(AVAILABLE_MLIPS) > 0
    assert "pet-omatpes" in AVAILABLE_MLIPS
    assert "pet-omatpes-v2" in AVAILABLE_MLIPS


def test_available_time_steps():
    """Test that AVAILABLE_TIME_STEPS is defined and contains expected values."""
    assert isinstance(AVAILABLE_TIME_STEPS, dict)
    assert len(AVAILABLE_TIME_STEPS) > 0

    # Check that each MLIP has time steps defined
    for mlip in AVAILABLE_MLIPS:
        assert mlip in AVAILABLE_TIME_STEPS
        assert isinstance(AVAILABLE_TIME_STEPS[mlip], list)
        assert len(AVAILABLE_TIME_STEPS[mlip]) > 0


def test_get_pretrained_invalid_mlip():
    """Test that get_pretrained raises ValueError for invalid MLIP."""
    from flashmd.models import get_pretrained

    with pytest.raises(ValueError, match="MLIP 'invalid_mlip' is not available"):
        get_pretrained(mlip="invalid_mlip", time_step=16)


def test_get_pretrained_invalid_time_step():
    """Test that get_pretrained raises ValueError for invalid time step."""
    from flashmd.models import get_pretrained

    with pytest.raises(ValueError, match="Pre-trained FlashMD models"):
        get_pretrained(mlip="pet-omatpes", time_step=999)


def test_save_checkpoint_invalid_time_step():
    """Test that save_checkpoint raises ValueError for invalid time step."""
    from flashmd.models import save_checkpoint

    with pytest.raises(ValueError, match="Pre-trained FlashMD models"):
        save_checkpoint(mlip="pet-omatpes-v2", time_step=999)


def test_save_checkpoint(monkeypatch):
    """Test that save_checkpoint saves the checkpoint file."""
    from flashmd.models import save_checkpoint

    # Mock hf_hub_download and shutil.copyfile
    def mock_hf_hub_download(repo_id, filename, cache_dir, revision):
        return f"/path/to/{filename}"

    def mock_copyfile(src, dst):
        assert src == "/path/to/flashmd_pet-omatpes-v2_16fs.ckpt"
        assert dst == "flashmd_pet-omatpes-v2_16fs.ckpt"

    monkeypatch.setattr("flashmd.models.hf_hub_download", mock_hf_hub_download)
    monkeypatch.setattr("flashmd.models.shutil.copyfile", mock_copyfile)

    save_checkpoint(mlip="pet-omatpes-v2", time_step=16)

