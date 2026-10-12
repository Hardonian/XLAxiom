"""Conservative cross-environment observation classification."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class VersionClassification(StrEnum):
    STABLE_PASS = "stable_pass"
    STABLE_FAILURE = "stable_failure"
    FIXED = "fixed"
    NEW_REGRESSION = "new_regression"
    CHANGED_BEHAVIOR = "changed_behavior"
    INCOMPARABLE = "incomparable"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


def classify_versions(older: dict[str, Any], newer: dict[str, Any]) -> dict[str, str]:
    """Classify two structured comparisons without asserting a root cause."""
    old_status, new_status = older.get("worker_status"), newer.get("worker_status")
    if old_status == "unsupported" or new_status == "unsupported":
        return {"classification": VersionClassification.UNSUPPORTED.value, "reason": "environment unsupported"}
    if old_status != "ok" or new_status != "ok":
        return {"classification": VersionClassification.INCONCLUSIVE.value, "reason": "environment execution incomplete"}
    old = older.get("comparison", {}).get("classification")
    new = newer.get("comparison", {}).get("classification")
    if old is None or new is None:
        return {"classification": VersionClassification.INCOMPARABLE.value, "reason": "missing comparison"}
    if old == "pass" and new == "pass":
        return {"classification": VersionClassification.STABLE_PASS.value, "reason": "both passed"}
    if old != "pass" and new == "pass":
        return {"classification": VersionClassification.FIXED.value, "reason": "observation disappeared"}
    if old == "pass" and new != "pass":
        return {"classification": VersionClassification.NEW_REGRESSION.value, "reason": "new observation"}
    if old == new:
        return {"classification": VersionClassification.STABLE_FAILURE.value, "reason": "same observation class"}
    return {"classification": VersionClassification.CHANGED_BEHAVIOR.value, "reason": "observation class changed"}
