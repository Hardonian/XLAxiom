import pytest

from xlaxiom.performance import PerformanceArtifactError, compare_benchmarks


def _record(median_ms: float) -> dict[str, object]:
    return {
        "backend": "jax-cpu-jit",
        "workload_sha256": "a" * 64,
        "measurements": 5,
        "median_ms": median_ms,
    }


def test_matching_benchmarks_classify_a_real_regression() -> None:
    assert compare_benchmarks(_record(10), _record(11))["classification"] == "pass"
    assert (
        compare_benchmarks(_record(10), _record(13))["classification"] == "performance_regression"
    )


def test_benchmarks_fail_closed_when_not_comparable_or_invalid() -> None:
    changed = {**_record(10), "workload_sha256": "b" * 64}
    assert compare_benchmarks(_record(10), changed)["classification"] == "inconclusive"
    with pytest.raises(PerformanceArtifactError, match="invalid median_ms"):
        compare_benchmarks(_record(0), _record(1))
