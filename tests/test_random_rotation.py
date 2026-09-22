import numpy as np

from flashmd.ase.velocity_verlet import _get_random_rotation


def _is_valid_rotation_matrix(R):
    return np.allclose(R @ R.T, np.eye(3)) and np.isclose(abs(np.linalg.det(R)), 1.0)


def test_default_rng_returns_valid_matrix():
    R = _get_random_rotation()
    assert R.shape == (3, 3)
    assert _is_valid_rotation_matrix(R)


def test_seeded_rng_is_reproducible():
    R1 = _get_random_rotation(np.random.default_rng(0))
    R2 = _get_random_rotation(np.random.default_rng(0))
    assert np.array_equal(R1, R2)
    assert _is_valid_rotation_matrix(R1)


def test_different_seeds_give_different_matrices():
    R1 = _get_random_rotation(np.random.default_rng(0))
    R2 = _get_random_rotation(np.random.default_rng(1))
    assert not np.array_equal(R1, R2)
