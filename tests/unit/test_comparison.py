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


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_signed_integer_minimum_compares_equal_without_absolute_value_overflow(dtype) -> None:
    """The signed minimum has no positive counterpart in its native integer dtype."""
    values = np.array([np.iinfo(dtype).min], dtype=dtype)

    result = compare(_ok((values,)), _ok((values.copy(),)), ComparisonPolicy())

    assert result.classification == Classification.PASS
    assert result.metrics[0].finite_mismatches == 0
    assert result.metrics[0].max_absolute_error == 0.0


def test_int64_extreme_difference_is_not_accepted_after_subtraction_wraparound() -> None:
    """A two-unit tolerance cannot hide the distance between the int64 extrema."""
    observed = np.array([np.iinfo(np.int64).min], dtype=np.int64)
    reference = np.array([np.iinfo(np.int64).max], dtype=np.int64)

    result = compare(
        _ok((observed,)),
        _ok((reference,)),
        ComparisonPolicy(rtol=0.0, atol=2.0),
    )

    assert result.classification == Classification.NUMERICAL_DISCREPANCY
    assert result.metrics[0].finite_mismatches == 1
    assert result.metrics[0].max_absolute_error > 1e18


def test_benchmark_jax_rejects_a_zero_measurement_budget() -> None:
    """``measurements=0`` previously raised an opaque ZeroDivisionError from the mean
    calculation. The contract is a clear ValueError before any work starts."""
    from xlaxiom.generation import synthetic_control

    with pytest.raises(ValueError, match="measurements must be at least 1"):
        benchmark_jax(synthetic_control(), measurements=0)


def test_output_metrics_reports_mean_and_rms_error() -> None:
    observed = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    reference = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    result = compare(_ok((observed,)), _ok((reference,)), ComparisonPolicy())
    metric = result.metrics[0]
    assert metric.max_absolute_error == pytest.approx(4.0)
    assert metric.mean_absolute_error == pytest.approx(2.0)
    assert metric.rms_error == pytest.approx(np.sqrt(20.0 / 3.0))


def test_benchmark_jax_returns_rich_percentiles_and_identity() -> None:
    from xlaxiom.generation import synthetic_control

    record = benchmark_jax(synthetic_control(), measurements=3)
    assert "workload_sha256" in record
    assert "min_ms" in record
    assert "max_ms" in record
    assert "p90_ms" in record
    assert "p95_ms" in record
    min_ms = float(record["min_ms"])
    median_ms = float(record["median_ms"])
    max_ms = float(record["max_ms"])
    assert min_ms <= median_ms <= max_ms
