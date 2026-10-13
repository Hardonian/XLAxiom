"""Bounded, deterministic CPU discovery campaigns over trusted generated workloads."""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

from .comparison import Classification
from .generation import GenerationConfig, generate
from .pipeline import run_comparison


@dataclass(frozen=True)
class CampaignConfig:
    cases: int = 20
    seed: int = 0
    max_duration_s: float = 180.0

    def validate(self) -> None:
        if not 1 <= self.cases <= 500:
            raise ValueError("cases must be between 1 and 500")
        if not 1 <= self.max_duration_s <= 900:
            raise ValueError("max_duration_s must be between 1 and 900")


def run_cpu_campaign(config: CampaignConfig) -> dict[str, Any]:
    """Compare NumPy/JAX JIT and JAX eager/JIT under an explicit campaign budget."""
    config.validate()
    started = time.perf_counter()
    counts: dict[str, int] = {classification.value: 0 for classification in Classification}
    eager_jit_counts: dict[str, int] = {
        classification.value: 0 for classification in Classification
    }
    anomalies: list[dict[str, Any]] = []
    completed = 0
    stopped = "completed"
    for offset in range(config.cases):
        if time.perf_counter() - started > config.max_duration_s:
            stopped = "duration_budget_exhausted"
            break
        workload = generate(GenerationConfig(seed=config.seed + offset))
        numpy_jit = run_comparison(workload, "jax-jit")
        eager_jit = run_comparison(workload, "jax-eager-vs-jit")
        counts[numpy_jit.classification.value] += 1
        eager_jit_counts[eager_jit.classification.value] += 1
        completed += 1
        if (
            numpy_jit.classification != Classification.PASS
            or eager_jit.classification != Classification.PASS
        ):
            anomalies.append(
                {
                    "seed": config.seed + offset,
                    "workload_sha256": workload.digest(),
                    "numpy_vs_jit": numpy_jit.to_dict(),
                    "eager_vs_jit": eager_jit.to_dict(),
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
        "warning": (
            "Generated workloads are trusted bounded graphs; persisted untrusted cases use workers."
        ),
    }
