import numpy as np
import pytest

from xlaxiom.comparison import (
    Classification,
    _max_or_zero,
    _max_relative_error,
    compare,
)
from xlaxiom.execution import ExecutionResult, ExecutionStatus, benchmark_jax
from xlaxiom.model import ComparisonPolicy


def test_nan_and_signed_zero_semantics_are_explicit() -> None:
    policy = ComparisonPolicy(atol=0, rtol=0, check_signed_zero=True)
    left = ExecutionResult(
        "a", "eager", ExecutionStatus.OK, (np.array([np.nan, -0.0], dtype=np.float32),)
    )
    right = ExecutionResult(
        "b", "eager", ExecutionStatus.OK, (np.array([np.nan, 0.0], dtype=np.float32),)
    )
    result = compare(left, right, policy)
    assert result.classification == Classification.NUMERICAL_DISCREPANCY
    assert result.metrics[0].signed_zero_mismatches == 1


def _ok(outputs: tuple[np.ndarray, ...]) -> ExecutionResult:
    return ExecutionResult("a", "eager", ExecutionStatus.OK, outputs)


def test_relative_error_is_scaled_by_reference_magnitude() -> None:
    """A small absolute difference against a zero reference must not read as a huge
    relative error. Scaling by an ``atol`` floor reported 10,000% for a 1e-7 delta,
    which is not interpretable."""
    policy = ComparisonPolicy(rtol=1e-3, atol=1e-9)
    observed = _ok((np.array([1e-7, 0.0], dtype=np.float64),))
    reference = _ok((np.array([0.0, 0.0], dtype=np.float64),))

    metrics = compare(observed, reference, policy).metrics[0]

    assert metrics.max_absolute_error == pytest.approx(1e-7)
    # Zero reference magnitude contributes no relative error; the absolute metric
    # is the honest signal for that element.
    assert metrics.max_relative_error == 0.0


def test_relative_error_reports_the_true_ratio_for_nonzero_references() -> None:
    policy = ComparisonPolicy(rtol=1e-3, atol=1e-9)
    observed = _ok((np.array([2.0, 0.5], dtype=np.float64),))
    reference = _ok((np.array([1.0, 1.0], dtype=np.float64),))

    metrics = compare(observed, reference, policy).metrics[0]

    assert metrics.max_relative_error == pytest.approx(1.0)


def test_max_or_zero_tolerates_an_empty_output() -> None:
    assert _max_or_zero(np.zeros((0,), dtype=np.float64)) == 0.0
    assert _max_or_zero(np.array([3.0, 9.0])) == 9.0


def test_max_relative_error_handles_zero_and_nonzero_magnitudes() -> None:
    absolute = np.array([1e-7, 0.5])
    expected = np.array([0.0, 1.0])
    assert _max_relative_error(absolute, expected) == pytest.approx(0.5)


def test_classification_is_unchanged_by_the_metric_redefinition() -> None:
    """The mismatch decision still uses the absolute tolerance policy, so the metric
    change cannot silently alter what counts as a discrepancy."""
    policy = ComparisonPolicy(rtol=1e-3, atol=1e-3)
    observed = _ok((np.array([2e-3], dtype=np.float64),))
    reference = _ok((np.array([0.0], dtype=np.float64),))

    result = compare(observed, reference, policy)

    assert result.classification == Classification.NUMERICAL_DISCREPANCY
    assert result.metrics[0].finite_mismatches == 1


def test_benchmark_jax_rejects_a_zero_measurement_budget() -> None:
    """``measurements=0`` previously raised an opaque ZeroDivisionError from the mean
    calculation. The contract is a clear ValueError before any work starts."""
    from xlaxiom.generation import synthetic_control

    with pytest.raises(ValueError, match="measurements must be at least 1"):
        benchmark_jax(synthetic_control(), measurements=0)
