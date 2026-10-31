"""Unit tests for automated tolerance calibration."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.calibration import calibrate_tolerances


def test_calibrate_tolerances_deterministic() -> None:
    def fn(x: np.ndarray) -> np.ndarray:
        return x * 2.0

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cal = calibrate_tolerances(fn, arr, runs=5)

    assert cal.is_deterministic is True
    assert cal.observed_max_abs_diff == 0.0
    assert cal.observed_max_rel_diff == 0.0
    assert cal.runs_evaluated == 5
    assert cal.recommended_rtol == 1e-5
    assert cal.recommended_atol == 1e-8
    assert "Deterministic" in cal.format_summary()


def test_calibrate_tolerances_stochastic() -> None:
    counter = 0

    def noisy_fn(x: np.ndarray) -> np.ndarray:
        nonlocal counter
        counter += 1
        # Add deterministic small perturbation across calls
        return x + (counter * 1e-4)

    arr = np.ones((5,), dtype=np.float32)
    cal = calibrate_tolerances(noisy_fn, arr, runs=5, safety_factor=3.0)

    assert cal.is_deterministic is False
    assert cal.observed_max_abs_diff > 0.0
    assert cal.observed_max_rel_diff > 0.0
    assert cal.recommended_atol >= cal.observed_max_abs_diff * 3.0
    assert "Stochastic" in cal.format_summary()


def test_calibrate_tolerances_invalid_runs() -> None:
    def fn(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="At least 2 runs are required"):
        calibrate_tolerances(fn, np.array([1.0]), runs=1)

