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


def _status_classification(result: ExecutionResult) -> Classification:
    return {
        ExecutionStatus.INVALID_WORKLOAD: Classification.INVALID_WORKLOAD,
        ExecutionStatus.UNSUPPORTED_BACKEND: Classification.UNSUPPORTED_BACKEND,
        ExecutionStatus.COMPILATION_FAILURE: Classification.COMPILATION_FAILURE,
        ExecutionStatus.EXECUTION_FAILURE: Classification.EXECUTION_FAILURE,
    }.get(result.status, Classification.INCONCLUSIVE)


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
        finite = np.isfinite(actual) & np.isfinite(expected)
        delta = np.zeros(actual.shape, dtype=np.float64)
        np.subtract(actual, expected, out=delta, where=finite, casting="unsafe")
        absolute = np.abs(delta)
        denominator = np.maximum(np.abs(expected), policy.atol)
        relative = np.zeros(actual.shape, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(absolute, denominator, out=relative, where=finite)
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
                float(np.max(absolute)),
                float(np.max(relative)),
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
