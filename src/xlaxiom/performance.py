"""Portable, conservative comparison of bounded benchmark evidence records."""

from __future__ import annotations

import math
from typing import Any

from .comparison import Classification


class PerformanceArtifactError(ValueError):
    """A performance record is incomplete or cannot be compared fairly."""


def compare_benchmarks(
    baseline: dict[str, Any], candidate: dict[str, Any], max_relative_regression: float = 0.2
) -> dict[str, Any]:
    """Classify a regression only for matching, bounded benchmark identities."""
    if not 0 <= max_relative_regression <= 10:
        raise PerformanceArtifactError("max_relative_regression must be between 0 and 10")
    for name, record in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(record.get("backend"), str) or not isinstance(
            record.get("workload_sha256"), str
        ):
            raise PerformanceArtifactError(f"{name} benchmark lacks backend or workload identity")
        value = record.get("median_ms")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise PerformanceArtifactError(f"{name} benchmark has invalid median_ms")
    compatible = (
        baseline["backend"] == candidate["backend"]
        and baseline["workload_sha256"] == candidate["workload_sha256"]
        and baseline.get("measurements") == candidate.get("measurements")
    )
    if not compatible:
        return {
            "classification": Classification.INCONCLUSIVE.value,
            "reason": "benchmark backend, workload identity, or sample count differs",
            "comparable": False,
        }
    ratio = float(candidate["median_ms"]) / float(baseline["median_ms"])
    regression = ratio - 1.0
    return {
        "classification": (
            Classification.PERFORMANCE_REGRESSION.value
            if regression > max_relative_regression
            else Classification.PASS.value
        ),
        "comparable": True,
        "baseline_median_ms": baseline["median_ms"],
        "candidate_median_ms": candidate["median_ms"],
        "ratio": ratio,
        "relative_regression": regression,
        "max_relative_regression": max_relative_regression,
        "reason": (
            "candidate median exceeds the configured regression budget"
            if regression > max_relative_regression
            else "candidate median is within the configured regression budget"
        ),
    }
