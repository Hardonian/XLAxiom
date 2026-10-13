"""Execution, minimization, and fresh-process replay for relational artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .comparison import Classification, ComparisonResult, compare
from .execution import execute_jax, execute_numpy
from .minimization import MinimizationStats, minimize
from .model import Workload
from .storage import ArtifactError, RelationalArtifact, atomic_write_text, save_json


def run_relational_comparison(artifact: RelationalArtifact) -> ComparisonResult:
    """Evaluate both sides of a stored equality relation on one explicit backend."""
    if artifact.backend == "numpy":
        left = execute_numpy(artifact.left)
        right = execute_numpy(artifact.right)
    elif artifact.backend == "jax-eager":
        left = execute_jax(artifact.left, requested_platform="cpu")
        right = execute_jax(artifact.right, requested_platform="cpu")
    elif artifact.backend == "jax-jit":
        left = execute_jax(artifact.left, jit=True, requested_platform="cpu")
        right = execute_jax(artifact.right, jit=True, requested_platform="cpu")
    else:
        raise ArtifactError(f"unsupported relational backend {artifact.backend!r}")
    return compare(left, right, artifact.left.policy)


@dataclass(frozen=True)
class RelationalMinimizationResult:
    artifact: RelationalArtifact
    left: MinimizationStats
    right: MinimizationStats
    duration_ms: float


def minimize_relational(
    artifact: RelationalArtifact, max_evaluations: int = 64
) -> RelationalMinimizationResult:
    """Greedily shrink each side while preserving the exact relational signature."""
    baseline = run_relational_comparison(artifact)
    expected = artifact.expected_classification
    if expected is None or expected == Classification.PASS.value:
        expected = baseline.classification.value
    if baseline.classification.value != expected:
        raise ArtifactError("relational failure predicate is not currently reproducible")
    if baseline.classification == Classification.PASS:
        raise ArtifactError("relational minimization requires a currently failing relation")
    signature = baseline.failure_signature()
    started = perf_counter()

    def left_predicate(candidate: Workload) -> str:
        return run_relational_comparison(replace(artifact, left=candidate)).failure_signature()

    left_result = minimize(artifact.left, left_predicate, signature, max_evaluations)
    left_artifact = replace(artifact, left=left_result.workload)

    def right_predicate(candidate: Workload) -> str:
        return run_relational_comparison(replace(left_artifact, right=candidate)).failure_signature()

    right_result = minimize(
        left_artifact.right,
        right_predicate,
        signature,
        max_evaluations,
    )
    minimized = replace(
        left_artifact,
        right=right_result.workload,
        expected_classification=expected,
        minimization={
            "original_left_sha256": artifact.left.digest(),
            "original_right_sha256": artifact.right.digest(),
            "left": asdict(left_result.stats),
            "right": asdict(right_result.stats),
            "preserved_signature": signature,
        },
    )
    return RelationalMinimizationResult(
        minimized,
        left_result.stats,
        right_result.stats,
        (perf_counter() - started) * 1000,
    )


def reproduce_relational_isolated(
    artifact: RelationalArtifact, attempts: int = 2, timeout_s: float = 60.0
) -> dict[str, Any]:
    """Re-run a relational predicate in fresh interpreters without shared JAX state."""
    if attempts < 1 or timeout_s <= 0:
        raise ArtifactError("attempts and timeout_s must be positive")
    script = (
        "import json,sys\n"
        "from xlaxiom.relational import run_relational_comparison\n"
        "from xlaxiom.storage import RelationalArtifact\n"
        "artifact = RelationalArtifact.from_dict(json.loads(sys.argv[1]))\n"
        "print(json.dumps(run_relational_comparison(artifact).to_dict(), sort_keys=True))\n"
    )
    observations: list[dict[str, Any]] = []
    for _ in range(attempts):
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, json.dumps(artifact.to_dict())],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            observations.append({"state": "blocked", "error": repr(exc)})
            continue
        if completed.returncode:
            observations.append({"state": "blocked", "error": completed.stderr[-500:]})
            continue
        try:
            payload = json.loads(completed.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            observations.append({"state": "inconclusive", "error": "no comparison record"})
            continue
        observations.append({"state": "observed", "classification": payload["classification"]})
    classifications = {
        str(item["classification"]) for item in observations if item["state"] == "observed"
    }
    reproduced = len(observations) == attempts and len(classifications) == 1
    return {
        "attempts": attempts,
        "observations": observations,
        "reproduced": reproduced,
        "classification": next(iter(classifications)) if reproduced else None,
        "note": "Each attempt ran in a fresh interpreter with no shared JAX compilation cache.",
    }


def build_relational_report(
    artifact: RelationalArtifact, comparison: ComparisonResult
) -> dict[str, Any]:
    """Build a report that identifies both sides of an equality relation."""
    return {
        "report_version": "1.0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "artifact_kind": "relational-case",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "observed_result": {
            "artifact_sha256": artifact.to_dict()["sha256"],
            "identifier": artifact.identifier,
            "relation": artifact.relation,
            "provenance": artifact.provenance,
            "left_workload_sha256": artifact.left.digest(),
            "right_workload_sha256": artifact.right.digest(),
            "comparison": comparison.to_dict(),
        },
        "interpretation": (
            "A relational discrepancy is evidence that this bounded relation did not hold under "
            "the recorded policy. It does not identify a root cause or a faulty backend."
        ),
        "upstream_candidate": comparison.classification == Classification.NUMERICAL_DISCREPANCY,
        "issue_draft": {
            "title": f"Relational observation: {comparison.classification.value} for {artifact.identifier}",
            "reproduction": "xlaxiom relation reproduce --case <relation.json>",
            "do_not_file_automatically": True,
        },
    }


def write_relational_report(
    directory: str | Path, artifact: RelationalArtifact, comparison: ComparisonResult
) -> tuple[Path, Path]:
    """Write checksummed-pair evidence in JSON and a concise reviewer-facing summary."""
    destination = Path(directory)
    report = build_relational_report(artifact, comparison)
    json_path = save_json(destination / "report.json", report)
    markdown = "\n".join(
        [
            "# XLAxiom relational evidence report",
            "",
            "## Observed relation",
            "",
            f"- Identifier: `{artifact.identifier}`",
            f"- Relation: `{artifact.relation}`",
            f"- Classification: `{comparison.classification.value}`",
            f"- Left workload: `{artifact.left.digest()}`",
            f"- Right workload: `{artifact.right.digest()}`",
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Reproduction",
            "",
            "```console",
            "xlaxiom relation reproduce --case relation.json",
            "```",
            "",
            "No upstream issue is filed by XLAxiom.",
            "",
        ]
    )
    markdown_path = destination / "report.md"
    atomic_write_text(markdown_path, markdown)
    return json_path, markdown_path
