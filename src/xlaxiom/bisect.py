"""Automated compiler version bisection for regression isolation.

Pinpoints the exact compiler version or release where a numerical divergence
or failure was first introduced using isolated uv virtual environments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .matrix import MatrixEnvironment, _run_environment, validate_jax_version


@dataclass(frozen=True)
class BisectStep:
    version: str
    index: int
    status: str
    classification: str
    duration_s: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BisectResult:
    case_path: str
    total_versions_considered: int
    steps_executed: int
    tested_steps: tuple[BisectStep, ...]
    first_bad_version: str | None
    last_good_version: str | None
    reproduction_command: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_path": self.case_path,
            "total_versions_considered": self.total_versions_considered,
            "steps_executed": self.steps_executed,
            "first_bad_version": self.first_bad_version,
            "last_good_version": self.last_good_version,
            "reproduction_command": self.reproduction_command,
            "steps": [step.to_dict() for step in self.tested_steps],
        }


def _evaluate_version(
    version: str,
    index: int,
    case_path: Path,
    timeout_s: float,
) -> BisectStep:
    validate_jax_version(version)
    env = MatrixEnvironment(name=f"bisect-{version}", jax=version, timeout_s=timeout_s)
    run_record = _run_environment(env, case_path)

    worker_status = run_record.get("worker_status", "failed")
    duration = float(run_record.get("duration_s", 0.0))

    if worker_status == "ok":
        raw_output = run_record.get("result", {})
        classification = raw_output.get("classification", "pass")
        error = None
    else:
        classification = "error"
        error = run_record.get("stderr") or run_record.get("stdout")

    return BisectStep(
        version=version,
        index=index,
        status=worker_status,
        classification=classification,
        duration_s=duration,
        error=error[:500] if error else None,
    )


def bisect_versions(
    case_path: str | Path,
    versions: Sequence[str],
    timeout_s: float = 90.0,
    evaluator: Any | None = None,
) -> BisectResult:
    """Perform binary search across ordered compiler versions to find a regression point."""
    path = Path(case_path)
    if not path.is_file():
        raise FileNotFoundError(f"case file does not exist: {path}")

    cleaned_versions = [validate_jax_version(v.strip()) for v in versions if v.strip()]
    if len(cleaned_versions) < 2:
        raise ValueError("bisection requires at least two versions to compare")

    eval_fn = evaluator or _evaluate_version
    tested_map: dict[int, BisectStep] = {}

    def test_idx(idx: int) -> BisectStep:
        if idx not in tested_map:
            ver = cleaned_versions[idx]
            step = eval_fn(ver, idx, path, timeout_s)
            tested_map[idx] = step
        return tested_map[idx]

    # Test endpoints first
    first_step = test_idx(0)
    last_step = test_idx(len(cleaned_versions) - 1)

    first_is_pass = first_step.classification == "pass"
    last_is_pass = last_step.classification == "pass"

    if first_is_pass == last_is_pass:
        return BisectResult(
            case_path=str(path),
            total_versions_considered=len(cleaned_versions),
            steps_executed=len(tested_map),
            tested_steps=tuple(tested_map.values()),
            first_bad_version=None if first_is_pass else cleaned_versions[0],
            last_good_version=cleaned_versions[0] if first_is_pass else None,
            reproduction_command=f"xlaxiom run --case {path}",
        )

    # Standard binary search: low is good, high is bad
    low = 0
    high = len(cleaned_versions) - 1

    while high - low > 1:
        mid = (low + high) // 2
        step = test_idx(mid)
        if step.classification == "pass":
            low = mid
        else:
            high = mid

    last_good = cleaned_versions[low]
    first_bad = cleaned_versions[high]

    return BisectResult(
        case_path=str(path),
        total_versions_considered=len(cleaned_versions),
        steps_executed=len(tested_map),
        tested_steps=tuple(tested_map.values()),
        first_bad_version=first_bad,
        last_good_version=last_good,
        reproduction_command=f"uv run --isolated --with jax=={first_bad} xlaxiom run --case {path}",
    )
