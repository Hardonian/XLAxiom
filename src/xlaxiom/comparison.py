"""Explicit numerical policies and discrepancy classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .execution import ExecutionResult, ExecutionStatus
from .model import ComparisonPolicy


class Classification(StrEnum):
    PASS = "pass"
    INVALID_WORKLOAD = "invalid_workload"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    ENVIRONMENT_FAILURE = "environment_failure"
    COMPILATION_FAILURE = "compilation_failure"
    EXECUTION_FAILURE = "execution_failure"
    TIMEOUT = "timeout"
    SHAPE_MISMATCH = "shape_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    NUMERICAL_DISCREPANCY = "numerical_discrepancy"
    NONDETERMINISM = "nondeterminism"
    PERFORMANCE_REGRESSION = "performance_regression"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class OutputMetrics:
    shape: tuple[int, ...]
    dtype: str
    max_absolute_error: float
    max_relative_error: float
    finite_mismatches: int
    nonfinite_mismatches: int
    signed_zero_mismatches: int


@dataclass(frozen=True)
class ComparisonResult:
    classification: Classification
    explanation: str
    policy: ComparisonPolicy
    metrics: tuple[OutputMetrics, ...] = ()
    left: dict[str, Any] | None = None
    right: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["classification"] = self.classification.value
        return result

    def failure_signature(self) -> str:
        """A stable predicate stronger than a broad classification alone."""
        kinds = tuple(
            (
                index,
                metric.finite_mismatches > 0,
                metric.nonfinite_mismatches > 0,
                metric.signed_zero_mismatches > 0,
            )
            for index, metric in enumerate(self.metrics)
            if metric.finite_mismatches
            or metric.nonfinite_mismatches
            or metric.signed_zero_mismatches
        )
        return repr((self.classification.value, kinds))


def _status_classification(result: ExecutionResult) -> Classification:
    return {
        ExecutionStatus.INVALID_WORKLOAD: Classification.INVALID_WORKLOAD,
        ExecutionStatus.UNSUPPORTED_BACKEND: Classification.UNSUPPORTED_BACKEND,
        ExecutionStatus.COMPILATION_FAILURE: Classification.COMPILATION_FAILURE,
        ExecutionStatus.EXECUTION_FAILURE: Classification.EXECUTION_FAILURE,
        ExecutionStatus.DEVICE_MISMATCH: Classification.ENVIRONMENT_FAILURE,
    }.get(result.status, Classification.INCONCLUSIVE)


def _max_or_zero(values: Any) -> float:
    """Maximum absolute error, or 0.0 for an output with no elements.

    ``np.max`` raises on a zero-size array. An empty output is a legitimate,
    validated shape, so the metric reports zero rather than failing the comparison.
    """
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _max_relative_error(absolute: Any, expected: Any) -> float:
    """Worst-case relative error, scaled by each element's own magnitude.

    Dividing by an ``atol`` floor — as an earlier revision did — reports a relative
    error of 1.0 for a 1e-7 difference against a 1e-7 tolerance, which reads as a
    100% error and is not interpretable. Scaling by ``|expected|`` per element keeps
    the metric meaningful; elements whose reference magnitude is zero contribute
    nothing, and the absolute-error metric already covers those cases.
    """
    magnitude = np.abs(expected)
    relative = np.zeros(absolute.shape, dtype=np.float64)
    scalable = magnitude > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(absolute, magnitude, out=relative, where=scalable)
    return _max_or_zero(relative)


def _integer_order(values: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """Map one integer dtype onto ascending ``uint64`` values without overflow."""
    if values.dtype.kind == "u":
        return values.astype(np.uint64, copy=False)
    bits = values.dtype.itemsize * 8
    unsigned_dtype = np.dtype(f"uint{bits}")
    # Flipping the sign bit turns two's-complement signed order into unsigned order:
    # INT_MIN becomes zero, zero becomes the sign bit, and INT_MAX becomes all ones.
    sign_bit = np.array(1 << (bits - 1), dtype=unsigned_dtype)
    return np.bitwise_xor(values.view(unsigned_dtype), sign_bit).astype(np.uint64)


def _integer_absolute_difference(
    actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]
) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """Return exact integer distances, including the ``int64`` extrema gap."""
    actual_order = _integer_order(actual)
    expected_order = _integer_order(expected)
    actual_is_larger = actual >= expected
    difference = np.empty(actual.shape, dtype=np.uint64)
    np.subtract(actual_order, expected_order, out=difference, where=actual_is_larger)
    np.subtract(expected_order, actual_order, out=difference, where=~actual_is_larger)
    return difference


def _integer_magnitude(values: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint64]]:
    """Return exact absolute values without overflowing at a signed minimum."""
    if values.dtype.kind == "u":
        return values.astype(np.uint64, copy=False)
    ordered = _integer_order(values)
    zero = np.uint64(1 << (values.dtype.itemsize * 8 - 1))
    nonnegative = values >= 0
    magnitude = np.empty(values.shape, dtype=np.uint64)
    np.subtract(ordered, zero, out=magnitude, where=nonnegative)
    np.subtract(zero, ordered, out=magnitude, where=~nonnegative)
    return magnitude


def _integer_metrics(
    actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any], policy: ComparisonPolicy
) -> OutputMetrics:
    """Compare integer outputs without native subtraction or absolute-value wrapping."""
    absolute = _integer_absolute_difference(actual, expected)
    magnitude = _integer_magnitude(expected)
    if policy.atol == 0 and policy.rtol == 0:
        close = actual == expected
    else:
        tolerance = policy.atol + policy.rtol * magnitude.astype(np.float64)
        close = absolute.astype(np.float64) <= tolerance
    relative = np.zeros(actual.shape, dtype=np.float64)
    scalable = magnitude > 0
    np.divide(
        absolute.astype(np.float64),
        magnitude.astype(np.float64),
        out=relative,
        where=scalable,
    )
    return OutputMetrics(
        tuple(actual.shape),
        str(actual.dtype),
        _max_or_zero(absolute),
        _max_or_zero(relative),
        int((~close).sum()),
        0,
        0,
    )


def compare(
    left: ExecutionResult, right: ExecutionResult, policy: ComparisonPolicy
) -> ComparisonResult:
    if left.status != ExecutionStatus.OK or right.status != ExecutionStatus.OK:
        failed = left if left.status != ExecutionStatus.OK else right
        return ComparisonResult(
            _status_classification(failed),
            f"{failed.backend}/{failed.mode} did not complete: {failed.status.value}",
            policy,
            left=left.summary(),
            right=right.summary(),
        )
    if len(left.outputs) != len(right.outputs):
        return ComparisonResult(Classification.SHAPE_MISMATCH, "different output arity", policy)
    metrics: list[OutputMetrics] = []
    for actual, expected in zip(left.outputs, right.outputs, strict=True):
        if actual.shape != expected.shape:
            return ComparisonResult(Classification.SHAPE_MISMATCH, "output shapes differ", policy)
        if actual.dtype != expected.dtype:
            return ComparisonResult(Classification.DTYPE_MISMATCH, "output dtypes differ", policy)
        if actual.dtype.kind in "iu":
            metrics.append(_integer_metrics(actual, expected, policy))
            continue
        finite = np.isfinite(actual) & np.isfinite(expected)
        delta = np.zeros(actual.shape, dtype=np.float64)
        np.subtract(actual, expected, out=delta, where=finite, casting="unsafe")
        absolute = np.abs(delta)
        close = absolute <= (policy.atol + policy.rtol * np.abs(expected))
        finite_mismatches = int((finite & ~close).sum())
        same_nan = (
            np.isnan(actual) & np.isnan(expected)
            if policy.equal_nan
            else np.zeros(actual.shape, bool)
        )
        same_inf = (
            np.isinf(actual) & np.isinf(expected) & (np.signbit(actual) == np.signbit(expected))
        )
        nonfinite = ~(finite | same_nan | same_inf)
        signed_zero = np.zeros(actual.shape, dtype=bool)
        if policy.check_signed_zero:
            signed_zero = (
                (actual == 0) & (expected == 0) & (np.signbit(actual) != np.signbit(expected))
            )
        metrics.append(
            OutputMetrics(
                tuple(actual.shape),
                str(actual.dtype),
                _max_or_zero(absolute),
                _max_relative_error(absolute, expected),
                finite_mismatches,
                int(nonfinite.sum()),
                int(signed_zero.sum()),
            )
        )
    has_discrepancy = any(
        metric.finite_mismatches or metric.nonfinite_mismatches or metric.signed_zero_mismatches
        for metric in metrics
    )
    return ComparisonResult(
        Classification.NUMERICAL_DISCREPANCY if has_discrepancy else Classification.PASS,
        "outputs differ under the configured policy"
        if has_discrepancy
        else "outputs agree under the configured policy",
        policy,
        tuple(metrics),
        left.summary(),
        right.summary(),
    )
