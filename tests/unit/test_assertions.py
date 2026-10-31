"""Unit tests for ergonomic numerical assertions and diagnostic diffing."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.assertions import (
    NumericalAssertionError,
    assert_close,
    check_parity,
    diff,
)


def test_diff_exact_match() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = diff(a, b)
    assert result.is_close is True
    assert result.mismatched_elements == 0
    assert result.max_absolute_error == 0.0
    assert result.cosine_similarity == 1.0
    assert "Exact match" in result.diagnosis


def test_diff_within_tolerance() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.00001, 1.99999, 3.00001], dtype=np.float32)
    result = diff(a, b, rtol=1e-4, atol=1e-4)
    assert result.is_close is True
    assert result.mismatched_elements == 0
    assert result.max_absolute_error < 1e-4


def test_diff_shape_mismatch() -> None:
    a = np.ones((2, 3))
    b = np.ones((3, 2))
    result = diff(a, b)
    assert result.is_close is False
    assert "Shape mismatch" in result.diagnosis


def test_diff_empty_arrays() -> None:
    a = np.empty((0, 4))
    b = np.empty((0, 4))
    result = diff(a, b)
    assert result.is_close is True
    assert result.total_elements == 0


def test_diff_catastrophic_cancellation() -> None:
    # High relative error with moderate absolute error
    a = np.array([1e-5], dtype=np.float64)
    b = np.array([2e-5], dtype=np.float64)
    result = diff(a, b, rtol=1e-5, atol=1e-8)
    assert result.is_close is False
    assert result.catastrophic_cancellations > 0
    assert "Catastrophic cancellation" in result.diagnosis


def test_diff_subnormal_underflow() -> None:
    a = np.array([0.0], dtype=np.float32)
    b = np.array([1e-38], dtype=np.float32)
    result = diff(a, b, rtol=1e-5, atol=1e-40)
    assert result.is_close is False
    assert result.subnormal_underflows > 0
    assert "Subnormal underflow" in result.diagnosis


def test_diff_signed_zero() -> None:
    a = np.array([0.0], dtype=np.float32)
    b = np.array([-0.0], dtype=np.float32)
    result_loose = diff(a, b, check_signed_zero=False)
    assert result_loose.is_close is True

    result_strict = diff(a, b, check_signed_zero=True)
    assert result_strict.is_close is False
    assert result_strict.sign_flips == 1
    assert "Signed zero" in result_strict.diagnosis


def test_diff_nan_and_inf() -> None:
    a = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    b = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    result = diff(a, b, equal_nan=True)
    assert result.is_close is True

    b_drift = np.array([1.0, np.inf, -np.inf], dtype=np.float32)
    res_drift = diff(a, b_drift, equal_nan=True)
    assert res_drift.is_close is False
    assert res_drift.nan_mismatches == 1


def test_assert_close_success() -> None:
    a = np.array([10.0, 20.0])
    b = np.array([10.0, 20.0])
    assert_close(a, b)


def test_assert_close_failure_raises_custom_error() -> None:
    a = np.array([10.0, 20.0])
    b = np.array([10.0, 25.0])
    with pytest.raises(NumericalAssertionError) as exc_info:
        assert_close(a, b, rtol=1e-5, atol=1e-5)
    assert exc_info.value.diff is not None
    assert exc_info.value.diff.is_close is False
    assert "Mismatched elements: 1 / 2" in str(exc_info.value)


def test_check_parity_functions() -> None:
    def fn_eager(x: np.ndarray) -> np.ndarray:
        return x * 2.0 + 1.0

    def fn_opt(x: np.ndarray) -> np.ndarray:
        return x * 2.0 + 1.0000001

    x = np.array([1.0, 2.0, 3.0])
    res = check_parity(fn_eager, fn_opt, x, rtol=1e-4, atol=1e-4)
    assert res.is_close is True
    assert len(res.diffs) == 1
    assert "Function Parity Verification: PASSED" in res.summary()


def test_repr_html() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 3.0])
    d = diff(a, b)
    html = d._repr_html_()
    assert "XLAxiom Numerical Verification" in html
    assert "DISCREPANCY" in html
