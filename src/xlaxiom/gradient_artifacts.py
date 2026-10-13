"""Validation and comparison for portable gradient-campaign evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GradientArtifactError(ValueError):
    """A gradient-campaign record cannot be safely compared."""


def load_gradient_campaign(path: str | Path) -> dict[str, Any]:
    """Load a campaign record with the minimum schema needed for comparison."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradientArtifactError(f"cannot read gradient campaign: {exc}") from exc
    if not isinstance(data, dict) or data.get("campaign_version") != "gradient-v1":
        raise GradientArtifactError("unsupported gradient campaign version")
    for key, expected in (
        ("configuration", dict),
        ("environment", dict),
        ("counts", dict),
        ("candidate_anomalies", list),
    ):
        if not isinstance(data.get(key), expected):
            raise GradientArtifactError(f"gradient campaign has invalid {key}")
    return data


def _signature(observation: object) -> str:
    if not isinstance(observation, dict):
        raise GradientArtifactError("candidate observation must be an object")
    case = observation.get("case")
    if not isinstance(case, dict):
        raise GradientArtifactError("candidate observation has no case")
    try:
        return ":".join(str(case[key]) for key in ("operation", "dtype", "value"))
    except KeyError as exc:
        raise GradientArtifactError(f"candidate observation missing {exc.args[0]}") from exc


def compare_gradient_campaigns(
    baseline: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, Any]:
    """Compare campaign configuration and distinct candidate case signatures."""
    baseline_configuration = baseline["configuration"]
    comparison_configuration = comparison["configuration"]
    if not isinstance(baseline_configuration, dict) or not isinstance(
        comparison_configuration, dict
    ):
        raise GradientArtifactError("gradient campaign has invalid configuration")
    configuration_match = baseline_configuration == comparison_configuration
    baseline_signatures = sorted({_signature(item) for item in baseline["candidate_anomalies"]})
    comparison_signatures = sorted({_signature(item) for item in comparison["candidate_anomalies"]})
    baseline_set = set(baseline_signatures)
    comparison_set = set(comparison_signatures)
    return {
        "comparison_version": "gradient-v1",
        "configuration_match": configuration_match,
        "baseline": {
            "configuration": baseline_configuration,
            "environment": baseline["environment"],
            "counts": baseline["counts"],
        },
        "comparison": {
            "configuration": comparison_configuration,
            "environment": comparison["environment"],
            "counts": comparison["counts"],
        },
        "candidate_signatures": {
            "shared": sorted(baseline_set & comparison_set),
            "baseline_only": sorted(baseline_set - comparison_set),
            "comparison_only": sorted(comparison_set - baseline_set),
        },
    }
