"""Unit tests for golden snapshot verification and pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xlaxiom.assertions import NumericalAssertionError
from xlaxiom.golden import check_golden, load_golden, save_golden


def test_golden_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "tensor.npz"
    actual = {"weight": np.ones((2, 3), dtype=np.float32), "bias": np.zeros(3, dtype=np.float32)}

    save_golden(path, actual)
    assert path.is_file()

    loaded = load_golden(path)
    assert set(loaded.keys()) == {"weight", "bias"}
    np.testing.assert_array_equal(loaded["weight"], actual["weight"])
    np.testing.assert_array_equal(loaded["bias"], actual["bias"])


def test_golden_check_passes(tmp_path: Path) -> None:
    path = tmp_path / "golden.npz"
    actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    save_golden(path, actual)
    # Check identical
    diffs = check_golden(path, actual)
    assert "data" in diffs
    assert diffs["data"].is_close


def test_golden_missing_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "nonexistent.npz"
    actual = np.ones(5)
    with pytest.raises(FileNotFoundError, match="Golden file not found"):
        check_golden(path, actual)


def test_golden_divergence_fails(tmp_path: Path) -> None:
    path = tmp_path / "golden.npz"
    desired = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    save_golden(path, desired)

    diverged = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    with pytest.raises(NumericalAssertionError, match="Golden comparison failed"):
        check_golden(path, diverged)


def test_golden_key_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "dict_golden.npz"
    save_golden(path, {"w1": np.ones(2), "w2": np.zeros(2)})

    with pytest.raises(NumericalAssertionError, match="Golden key mismatch"):
        check_golden(path, {"w1": np.ones(2)})


def test_golden_update_flag(tmp_path: Path) -> None:
    path = tmp_path / "golden.npz"
    v1 = np.array([1.0], dtype=np.float32)
    v2 = np.array([2.0], dtype=np.float32)

    save_golden(path, v1)
    # update=True rewrites the file
    check_golden(path, v2, update=True)

    loaded = load_golden(path)
    np.testing.assert_array_equal(loaded["data"], v2)


def test_golden_max_ulp_enforcement(tmp_path: Path) -> None:
    path = tmp_path / "golden.npz"
    v1 = np.array([1.0], dtype=np.float32)
    save_golden(path, v1)

    # Within rtol 0.1, but multiple ULPs away
    v2 = np.array([1.0001], dtype=np.float32)
    with pytest.raises(NumericalAssertionError, match="max ULP error"):
        check_golden(path, v2, rtol=1e-2, atol=1e-2, max_ulp=2.0)
