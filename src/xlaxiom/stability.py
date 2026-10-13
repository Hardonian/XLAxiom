"""Repeated-execution checks that distinguish a discrepancy from nondeterminism."""

from __future__ import annotations

from typing import Any

from .comparison import Classification, compare
from .model import Workload
from .pipeline import run_backend


def check_stability(workload: Workload, backend: str, runs: int = 3) -> dict[str, Any]:
    """Execute one backend repeatedly and preserve every pairwise observation."""
    if not 2 <= runs <= 50:
        raise ValueError("stability runs must be between 2 and 50")
    executions = [run_backend(workload, backend) for _ in range(runs)]
    baseline = executions[0]
    pairwise = [compare(result, baseline, workload.policy) for result in executions[1:]]
    classifications = [result.classification for result in pairwise]
    if all(classification == Classification.PASS for classification in classifications):
        classification = Classification.PASS
        explanation = "all repeated executions agree under the recorded policy"
    elif all(result.status.value == "ok" for result in executions):
        classification = Classification.NONDETERMINISM
        explanation = "repeated successful executions differ under the recorded policy"
    else:
        classification = next(item for item in classifications if item != Classification.PASS)
        explanation = "one or more repeated executions did not complete consistently"
    return {
        "evidence_version": "1.0",
        "kind": "stability-evidence",
        "workload_sha256": workload.digest(),
        "backend": backend,
        "runs": runs,
        "classification": classification.value,
        "explanation": explanation,
        "executions": [result.summary() for result in executions],
        "pairwise": [result.to_dict() for result in pairwise],
    }
