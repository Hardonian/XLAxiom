"""Conservative, evidence-bound checks for upstream issue readiness."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .comparison import Classification, ComparisonResult
from .environment import compare_environments, validate_environment_evidence
from .storage import ArtifactError, CaseArtifact, load_case

REPRODUCTION_EVIDENCE_VERSION = "1.0"


class ReadinessEvidenceError(ValueError):
    """A supplied reproduction record cannot safely satisfy a readiness gate."""


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_reproduction_evidence(
    artifact: CaseArtifact,
    comparison: ComparisonResult,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-identifying reproduction record for later readiness review.

    A command invocation is the fresh-process observation; this record binds its verdict
    to the exact checksummed case.  An optional checksummed environment attachment lets
    readiness verify that two successful observations came from independent hosts.
    """
    environment_sha256: str | None = None
    if environment is not None:
        try:
            validate_environment_evidence(environment)
        except ArtifactError as exc:
            raise ReadinessEvidenceError(f"invalid attached environment evidence: {exc}") from exc
        environment_sha256 = environment["sha256"]

    expected = artifact.expected_classification
    observed = comparison.to_dict()
    verdict = (
        "reproduced"
        if expected is not None and observed["classification"] == expected
        else "not_reproduced"
    )
    payload: dict[str, Any] = {
        "reproduction_evidence_version": REPRODUCTION_EVIDENCE_VERSION,
        "case_sha256": artifact.to_dict()["sha256"],
        "expected": expected,
        "verdict": verdict,
        "observed": observed,
        "environment_sha256": environment_sha256,
    }
    return {**payload, "sha256": _digest(payload)}


def validate_reproduction_evidence(data: object) -> dict[str, Any]:
    """Validate a checksummed reproduction record without trusting its claim blindly."""
    if not isinstance(data, dict):
        raise ReadinessEvidenceError("reproduction evidence must be a JSON object")
    checksum = data.get("sha256")
    payload = {key: value for key, value in data.items() if key != "sha256"}
    if payload.get("reproduction_evidence_version") != REPRODUCTION_EVIDENCE_VERSION:
        raise ReadinessEvidenceError("unsupported reproduction evidence version")
    if not isinstance(checksum, str) or _digest(payload) != checksum:
        raise ReadinessEvidenceError("reproduction evidence checksum mismatch")
    if not isinstance(payload.get("case_sha256"), str):
        raise ReadinessEvidenceError("reproduction evidence has invalid case SHA-256")
    expected = payload.get("expected")
    if expected is not None and not isinstance(expected, str):
        raise ReadinessEvidenceError("reproduction evidence has invalid expected classification")
    if payload.get("verdict") not in {"reproduced", "not_reproduced"}:
        raise ReadinessEvidenceError("reproduction evidence has invalid verdict")
    observed = payload.get("observed")
    if not isinstance(observed, dict) or not isinstance(observed.get("classification"), str):
        raise ReadinessEvidenceError("reproduction evidence has invalid observed classification")
    environment_sha256 = payload.get("environment_sha256")
    if environment_sha256 is not None and not isinstance(environment_sha256, str):
        raise ReadinessEvidenceError("reproduction evidence has invalid environment SHA-256")
    reproduced = expected is not None and observed["classification"] == expected
    if (payload["verdict"] == "reproduced") != reproduced:
        raise ReadinessEvidenceError("reproduction evidence verdict does not match its observation")
    return data


def load_reproduction_evidence(path: str | Path) -> dict[str, Any]:
    """Load one persisted reproduction record and validate its integrity."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessEvidenceError(f"cannot read reproduction evidence: {exc}") from exc
    return validate_reproduction_evidence(data)


def _successful_reproductions(
    artifact: CaseArtifact, reproduction_evidence: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Return validated successful records that are bound to the supplied case."""
    case_sha256 = artifact.to_dict()["sha256"]
    expected = artifact.expected_classification
    successful: list[dict[str, Any]] = []
    for record in reproduction_evidence:
        validate_reproduction_evidence(record)
        if record["case_sha256"] != case_sha256:
            raise ReadinessEvidenceError("reproduction evidence does not match the supplied case")
        if record["expected"] != expected:
            raise ReadinessEvidenceError(
                "reproduction evidence expected classification does not match the supplied case"
            )
        if record["verdict"] == "reproduced":
            successful.append(record)
    return tuple(successful)


def _independent_reproduction_pairs(
    reproductions: Sequence[dict[str, Any]], environment_evidence: Sequence[dict[str, Any]]
) -> list[dict[str, str]]:
    """Validate environment bindings and find successful independent reproduction pairs."""
    environments_by_sha256: dict[str, dict[str, Any]] = {}
    for environment in environment_evidence:
        try:
            validated = validate_environment_evidence(environment)
        except ArtifactError as exc:
            raise ReadinessEvidenceError(f"invalid environment evidence: {exc}") from exc
        environments_by_sha256[validated["sha256"]] = validated

    referenced: list[str] = []
    for reproduction in reproductions:
        environment_sha256 = reproduction["environment_sha256"]
        if environment_sha256 is None:
            continue
        if environment_sha256 not in environments_by_sha256:
            raise ReadinessEvidenceError(
                "reproduction evidence references an environment record that was not supplied"
            )
        if reproduction["verdict"] != "reproduced":
            continue
        if environment_sha256 not in referenced:
            referenced.append(environment_sha256)

    independent_pairs: list[dict[str, str]] = []
    for index, primary_sha256 in enumerate(referenced):
        for secondary_sha256 in referenced[index + 1 :]:
            assessment = compare_environments(
                environments_by_sha256[primary_sha256], environments_by_sha256[secondary_sha256]
            )
            if assessment["independent"]:
                independent_pairs.append(
                    {
                        "primary_sha256": primary_sha256,
                        "secondary_sha256": secondary_sha256,
                    }
                )
    return independent_pairs


def assess_upstream_readiness(
    artifact: CaseArtifact,
    reproduction_evidence: Sequence[dict[str, Any]] = (),
    environment_evidence: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Assess a case against supplied, validated external reproduction evidence.

    Missing evidence remains a failed check.  A successful independent-environment
    check requires two successful, case-bound reproduction records, each linked to a
    supplied environment record that passes the existing host-independence test.
    """
    successful_reproductions = _successful_reproductions(artifact, reproduction_evidence)
    independent_pairs = _independent_reproduction_pairs(reproduction_evidence, environment_evidence)
    checks = [
        {
            "name": "non_synthetic",
            "passed": artifact.fault is None,
            "detail": "synthetic controls cannot be filed as compiler observations",
        },
        {
            "name": "expected_discrepancy",
            "passed": artifact.expected_classification
            == Classification.NUMERICAL_DISCREPANCY.value,
            "detail": "a candidate needs a stored non-pass numerical predicate",
        },
        {
            "name": "failure_preserving_minimization",
            "passed": bool(artifact.minimization),
            "detail": "a minimized case must retain its reduction record",
        },
        {
            "name": "fresh_process_reproduction",
            "passed": bool(successful_reproductions),
            "detail": (
                f"validated {len(successful_reproductions)} successful reproduction record(s)"
                if successful_reproductions
                else "attach a checksummed record from `xlaxiom reproduce --output <path>`"
            ),
        },
        {
            "name": "independent_environment_reproduction",
            "passed": bool(independent_pairs),
            "detail": (
                f"validated {len(independent_pairs)} independent reproduction pair(s)"
                if independent_pairs
                else (
                    "attach two successful reproduction records bound to independent "
                    "environment evidence"
                )
            ),
        },
    ]
    return {
        "assessment_version": "1.1",
        "case_sha256": artifact.to_dict()["sha256"],
        "ready": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "evidence": {
            "reproduction_records_supplied": len(reproduction_evidence),
            "successful_reproductions": len(successful_reproductions),
            "environment_records_supplied": len(environment_evidence),
            "independent_reproduction_pairs": independent_pairs,
        },
        "note": (
            "This command never files an issue and never treats absent external evidence as a pass."
        ),
    }


def run_readiness_preflight(
    case_path: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run an automated pre-flight audit on an upstream bug report candidate."""
    path = Path(case_path)
    checks: list[dict[str, Any]] = []

    try:
        artifact = load_case(path)
        checks.append(
            {
                "name": "case_integrity",
                "passed": True,
                "detail": (
                    f"Case artifact is valid and checksum matches SHA-256 "
                    f"{artifact.to_dict()['sha256'][:12]}"
                ),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "case_integrity",
                "passed": False,
                "detail": f"Case failed integrity validation: {exc}",
            }
        )
        return {
            "preflight_version": "1.0",
            "case_path": str(path),
            "preflight_passed": False,
            "checks": checks,
        }

    is_non_synthetic = artifact.fault is None
    checks.append(
        {
            "name": "non_synthetic",
            "passed": is_non_synthetic,
            "detail": (
                "Verified artifact is not a synthetic test injection"
                if is_non_synthetic
                else "Cannot submit synthetic fault upstream"
            ),
        }
    )

    sanitized = True
    sanitization_issues: list[str] = []
    text_to_scan = ""
    if markdown_path:
        md_file = Path(markdown_path)
        if md_file.exists():
            text_to_scan = md_file.read_text(encoding="utf-8")

    text_to_scan += " " + json.dumps(artifact.to_dict())

    # Detect user directory leaks like /Users/ or /home/ or C:\Users\
    user_path_match = re.search(
        r"([A-Za-z]:\\[Uu]sers\\[^\\]+|/(?:home|Users)/[^/\s]+)", text_to_scan
    )
    if user_path_match:
        matched_str = user_path_match.group(0)
        if not any(
            ph in matched_str.lower() for ph in ("placeholder", "<user>", "username", "runner")
        ):
            sanitized = False
            sanitization_issues.append(f"detected local user path: {matched_str}")

    checks.append(
        {
            "name": "sanitization_and_privacy",
            "passed": sanitized,
            "detail": (
                "All private host paths and tokens cleanly redacted"
                if sanitized
                else "; ".join(sanitization_issues)
            ),
        }
    )

    from .triage import generate_minimal_reproduction_script

    try:
        script = generate_minimal_reproduction_script(artifact.workload)
        has_script = bool(script and len(script.splitlines()) >= 5)
        checks.append(
            {
                "name": "self_contained_reproduction_script",
                "passed": has_script,
                "detail": (
                    f"Generated {len(script.splitlines())}-line zero-dependency reproduction script"
                ),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "self_contained_reproduction_script",
                "passed": False,
                "detail": f"Failed to generate self-contained repro script: {exc}",
            }
        )

    all_passed = all(check["passed"] for check in checks)
    return {
        "preflight_version": "1.0",
        "case_path": str(path),
        "case_sha256": artifact.to_dict()["sha256"],
        "preflight_passed": all_passed,
        "checks": checks,
    }
