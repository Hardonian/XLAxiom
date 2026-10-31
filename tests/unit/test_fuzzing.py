"""Unit tests for the custom kernel differential fuzzer and shape minimizer."""

from __future__ import annotations

import numpy as np

from xlaxiom.fuzzing import fuzz_kernel


def test_fuzz_kernel_passing_case() -> None:
    # A correct kernel implementation matches reference on all shapes
    def dummy_kernel(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def ref_relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    # Dynamic shapes: B in [1, 4], S in [8, 32], D fixed to 16
    shape_spec = ((1, 4), (8, 32), 16)

    res = fuzz_kernel(
        dummy_kernel,
        ref_relu,
        shape_spec=shape_spec,
        num_trials=10,
        seed=101,
    )
    assert res.passed is True
    assert res.trials_run == 10
    assert "PASSED" in res.format_summary()


def test_fuzz_kernel_failing_boundary_detection() -> None:
    # Kernel fails specifically on prime sequence lengths > 17
    def flawed_kernel(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        if x.shape[1] > 17:
            # Bug: drops last element
            out[:, -1] += 0.5
        return out

    def ref_fn(x: np.ndarray) -> np.ndarray:
        return x.copy()

    shape_spec = (2, (16, 32))

    res = fuzz_kernel(
        flawed_kernel,
        ref_fn,
        shape_spec=shape_spec,
        num_trials=15,
        seed=42,
    )
    assert res.passed is False
    assert res.minimal_failing_shape is not None
    assert res.diff is not None
    assert res.repro_script is not None
    assert "Minimal reproducing shape" in res.repro_script
    assert "DISCREPANCY DETECTED" in res.format_summary()


def test_fuzz_kernel_exception_recovery() -> None:
    def crashing_kernel(x: np.ndarray) -> np.ndarray:
        if x.shape[0] > 1:
            raise RuntimeError("Out of bounds tile access simulated")
        return x

    def ref_fn(x: np.ndarray) -> np.ndarray:
        return x

    res = fuzz_kernel(
        crashing_kernel,
        ref_fn,
        shape_spec=((2, 4), 8),
        num_trials=5,
        seed=12,
    )
    assert res.passed is False
    assert "Kernel execution exception" in (res.repro_script or "")
