"""Bounded, deterministic CPU discovery campaigns over trusted generated workloads."""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from .comparison import Classification
from .generation import GenerationConfig, generate
from .pipeline import run_comparison


@dataclass(frozen=True)
class CampaignConfig:
    cases: int = 20
    seed: int = 0
    max_duration_s: float = 180.0
    isolate: bool = False
    case_timeout_s: float = 60.0
    case_memory_mb: int | None = None

    def validate(self) -> None:
        if not 1 <= self.cases <= 500:
            raise ValueError("cases must be between 1 and 500")
        if not 1 <= self.max_duration_s <= 900:
            raise ValueError("max_duration_s must be between 1 and 900")
        if self.isolate and not 1 <= self.case_timeout_s <= 600:
            raise ValueError("case_timeout_s must be between 1 and 600")
        if self.case_memory_mb is not None and not 16 <= self.case_memory_mb <= 65_536:
            raise ValueError("case_memory_mb must be between 16 and 65536 when supplied")


CAMPAIGN_PROFILES: dict[str, CampaignConfig] = {
    "smoke": CampaignConfig(
        cases=4,
        max_duration_s=60.0,
        isolate=True,
        case_timeout_s=30.0,
        case_memory_mb=1024,
    ),
    "standard": CampaignConfig(),
    "thorough": CampaignConfig(
        cases=100,
        max_duration_s=600.0,
        isolate=True,
        case_timeout_s=60.0,
        case_memory_mb=2048,
    ),
}


def campaign_profile(name: str) -> CampaignConfig:
    """Return a reviewed campaign budget; callers may override it explicitly."""
    try:
        return CAMPAIGN_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown campaign profile {name!r}") from exc


def configure_campaign(
    profile: str,
    *,
    cases: int | None = None,
    seed: int | None = None,
    max_duration_s: float | None = None,
    isolate: bool = False,
    case_timeout_s: float | None = None,
    case_memory_mb: int | None = None,
) -> CampaignConfig:
    """Apply explicit CLI overrides without obscuring the selected profile budget."""
    base = campaign_profile(profile)
    return replace(
        base,
        cases=base.cases if cases is None else cases,
        seed=base.seed if seed is None else seed,
        max_duration_s=base.max_duration_s if max_duration_s is None else max_duration_s,
        isolate=base.isolate or isolate,
        case_timeout_s=base.case_timeout_s if case_timeout_s is None else case_timeout_s,
        case_memory_mb=base.case_memory_mb if case_memory_mb is None else case_memory_mb,
    )


def _run_isolated(
    workload: Any, mode: str, config: CampaignConfig
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Run one comparison in a spawned child with a hard timeout.

    Returns the comparison record plus a diagnostic record when the child timed out or
    crashed. A contained failure is reported, never silently retried or treated as a
    pass.
    """
    from .safety import compare_isolated

    kwargs: dict[str, Any] = {"timeout_s": config.case_timeout_s}
    if config.case_memory_mb is not None:
        kwargs["memory_limit_mb"] = config.case_memory_mb
    result = compare_isolated(workload, mode, **kwargs)
    if result.status != "ok":
        return None, {"status": result.status, "error": result.error}
    return dict(result.record or {}), None


def run_cpu_campaign(config: CampaignConfig) -> dict[str, Any]:
    """Compare NumPy/JAX JIT and JAX eager/JIT under an explicit campaign budget.

    With ``isolate`` enabled each case runs in a spawned child under a hard timeout, so
    a hung kernel or an out-of-memory failure contains one case instead of taking down
    the whole campaign. Contained failures are recorded explicitly.
    """
    config.validate()
    started = time.perf_counter()
    counts: dict[str, int] = {classification.value: 0 for classification in Classification}
    eager_jit_counts: dict[str, int] = {
        classification.value: 0 for classification in Classification
    }
    anomalies: list[dict[str, Any]] = []
    contained: list[dict[str, Any]] = []
    completed = 0
    stopped = "completed"
    for offset in range(config.cases):
        if time.perf_counter() - started > config.max_duration_s:
            stopped = "duration_budget_exhausted"
            break
        workload = generate(GenerationConfig(seed=config.seed + offset))
        if config.isolate:
            numpy_jit, numpy_failure = _run_isolated(workload, "jax-jit", config)
            eager_jit, eager_failure = _run_isolated(workload, "jax-eager-vs-jit", config)
            if numpy_failure or eager_failure:
                contained.append(
                    {
                        "seed": config.seed + offset,
                        "workload_sha256": workload.digest(),
                        "numpy_vs_jit": numpy_failure,
                        "eager_vs_jit": eager_failure,
                    }
                )
        else:
            numpy_jit = run_comparison(workload, "jax-jit").to_dict()
            eager_jit = run_comparison(workload, "jax-eager-vs-jit").to_dict()
        if numpy_jit is not None:
            counts[str(numpy_jit["classification"])] += 1
        if eager_jit is not None:
            eager_jit_counts[str(eager_jit["classification"])] += 1
        completed += 1
        if (numpy_jit is not None and numpy_jit["classification"] != Classification.PASS.value) or (
            eager_jit is not None and eager_jit["classification"] != Classification.PASS.value
        ):
            anomalies.append(
                {
                    "seed": config.seed + offset,
                    "workload_sha256": workload.digest(),
                    "numpy_vs_jit": numpy_jit,
                    "eager_vs_jit": eager_jit,
                }
            )
    return {
        "campaign_version": "1.0",
        "configuration": asdict(config),
        "platform": platform.platform(),
        "requested_cases": config.cases,
        "completed_cases": completed,
        "stop_reason": stopped,
        "duration_s": time.perf_counter() - started,
        "numpy_vs_jit_counts": counts,
        "eager_vs_jit_counts": eager_jit_counts,
        "candidate_anomalies": anomalies,
        "contained_failures": contained,
        "isolation": {
            "enabled": config.isolate,
            "case_timeout_s": config.case_timeout_s if config.isolate else None,
            "case_memory_mb": config.case_memory_mb if config.isolate else None,
            "contained_count": len(contained),
        },
        "warning": (
            "Generated workloads are trusted bounded graphs; persisted untrusted cases use workers."
        ),
    }
