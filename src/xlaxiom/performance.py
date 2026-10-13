"""Portable, conservative comparison of bounded benchmark evidence records."""

from __future__ import annotations

import math
from typing import Any

from .comparison import Classification


class PerformanceArtifactError(ValueError):
    """A performance record is incomplete or cannot be compared fairly."""


def compare_compilation_cache(
    cold: dict[str, Any], warm: dict[str, Any], max_relative_warm_compile: float = 0.1
) -> dict[str, Any]:
    """Record a conservative same-process compilation-cache observation.

    A faster second compilation is evidence consistent with a cache; it is not proof of
    a particular JAX/OpenXLA cache implementation. A slower second compilation is
    inconclusive instead of being classified as a compiler defect.
    """
    if not 0 <= max_relative_warm_compile <= 10:
        raise PerformanceArtifactError("max_relative_warm_compile must be between 0 and 10")
    for name, record in (("cold", cold), ("warm", warm)):
        for field in ("backend", "workload_sha256", "compilation_ms"):
            if field not in record:
                raise PerformanceArtifactError(f"{name} benchmark lacks {field}")
        value = record["compilation_ms"]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise PerformanceArtifactError(f"{name} benchmark has invalid compilation_ms")
    if cold["backend"] != warm["backend"] or cold["workload_sha256"] != warm["workload_sha256"]:
        return {
            "classification": Classification.INCONCLUSIVE.value,
            "comparable": False,
            "reason": "benchmark backend or workload identity differs",
        }
    cold_ms = float(cold["compilation_ms"])
    warm_ms = float(warm["compilation_ms"])
    ratio = warm_ms / cold_ms if cold_ms else 1.0
    observed = warm_ms <= cold_ms * (1 + max_relative_warm_compile)
    return {
        "classification": Classification.PASS.value
        if observed
        else Classification.INCONCLUSIVE.value,
        "comparable": True,
        "cold_compilation_ms": cold_ms,
        "warm_compilation_ms": warm_ms,
        "ratio": ratio,
        "max_relative_warm_compile": max_relative_warm_compile,
        "observation": "warm_compile_not_slower" if observed else "warm_compile_slower",
        "reason": (
            "second same-process compilation is consistent with cache reuse"
            if observed
            else "second compilation was slower; cache behavior is inconclusive"
        ),
    }


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
