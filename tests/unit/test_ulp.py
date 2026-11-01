"""Unit tests for ULP distance and statistical analysis."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.assertions import NumericalAssertionError
from xlaxiom.ulp import (
    assert_ulp_close,
    max_ulp_error,
    ulp_distance,
    ulp_stats,
)


def test_ulp_exact_match() -> None:
    x = np.array([1.0, 2.0, -3.5, 0.0, -0.0], dtype=np.float32)
    dist = ulp_distance(x, x)
    assert np.all(dist == 0.0)
    assert max_ulp_error(x, x) == 0.0

    # Signed zero has distance 0 in IEEE 754 float representation
    pos_zero = np.array([0.0], dtype=np.float32)
    neg_zero = np.array([-0.0], dtype=np.float32)
    assert ulp_distance(pos_zero, neg_zero)[0] == 0.0


def test_ulp_adjacent_floats() -> None:
    val = np.float32(1.0)
    # The next representable float after 1.0f is 1.0 + 2^-23 (1.0000001192092896)
    next_val = np.nextafter(val, np.float32(2.0))
    dist = ulp_distance(np.array([val]), np.array([next_val]))
    assert dist[0] == 1.0

    # 4 steps away
    v4 = val
    for _ in range(4):
        v4 = np.nextafter(v4, np.float32(2.0))
    dist4 = ulp_distance(np.array([val]), np.array([v4]))
    assert dist4[0] == 4.0


def test_ulp_nan_handling() -> None:
    # Identical NaNs have distance 0.0
    a = np.array([np.nan, 1.0], dtype=np.float32)
    b = np.array([np.nan, 1.0], dtype=np.float32)
    dist = ulp_distance(a, b)
    assert dist[0] == 0.0
    assert dist[1] == 0.0

    # Mismatched NaN has distance inf
    c = np.array([2.0, 1.0], dtype=np.float32)
    dist_mismatch = ulp_distance(a, c)
    assert np.isinf(dist_mismatch[0])
    assert dist_mismatch[1] == 0.0


def test_ulp_stats_formatting() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = a.copy()
    b[0] = np.nextafter(a[0], np.float32(2.0))  # 1 ULP
    b[1] = np.nextafter(b[1], np.float32(3.0))  # 1 ULP

    stats = ulp_stats(b, a)
    assert stats.max_ulp == 1.0
    assert stats.total_elements == 4
    assert stats.exact_fraction == 0.5
    summary = stats.format_summary()
    assert "max=1" in summary
    assert "bit-exact=50.0%" in summary


def test_assert_ulp_close_passes() -> None:
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([np.nextafter(np.float32(1.0), np.float32(2.0)), np.float32(2.0)])
    assert_ulp_close(a, b, max_ulp=2.0)


def test_assert_ulp_close_fails() -> None:
    a = np.array([1.0], dtype=np.float32)
    b = np.array([1.5], dtype=np.float32)  # Millions of ULPs apart
    with pytest.raises(NumericalAssertionError, match="ULP assertion failed"):
        assert_ulp_close(a, b, max_ulp=4.0)


def test_ulp_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        ulp_distance(np.ones(2), np.ones(3))
