"""Conservative checks for whether a case is ready for human upstream review."""

from __future__ import annotations

from typing import Any

from .comparison import Classification
from .storage import CaseArtifact


def assess_upstream_readiness(artifact: CaseArtifact) -> dict[str, Any]:
    """Report evidence present in an artifact without inventing missing reproduction."""
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
            "passed": False,
            "detail": "run `xlaxiom reproduce --case <case.json>` and attach its verdict",
        },
        {
            "name": "independent_environment_reproduction",
            "passed": False,
            "detail": "attach a second environment/device evidence record before filing",
        },
    ]
    return {
        "assessment_version": "1.0",
        "case_sha256": artifact.to_dict()["sha256"],
        "ready": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "note": (
            "This command never files an issue and never treats absent external evidence as a pass."
        ),
    }
