import pytest

from xlaxiom.gradient_artifacts import GradientArtifactError, compare_gradient_campaigns


def _campaign(value: float, *, seed: int = 0) -> dict[str, object]:
    return {
        "campaign_version": "gradient-v1",
        "configuration": {"cases": 40, "seed": seed, "oracle": "analytic-v1"},
        "environment": {"devices": ["cpu:0"]},
        "counts": {"candidate_discrepancy": 1},
        "candidate_anomalies": [
            {"case": {"operation": "tanh", "dtype": "float32", "value": value}}
        ],
    }


def test_compares_signatures_and_environment_without_hardware_assumptions() -> None:
    baseline = _campaign(10.0)
    comparison = _campaign(1.0)
    result = compare_gradient_campaigns(baseline, comparison)
    assert result["configuration_match"] is True
    assert result["candidate_signatures"] == {
        "shared": [],
        "baseline_only": ["tanh:float32:10.0"],
        "comparison_only": ["tanh:float32:1.0"],
    }


def test_rejects_missing_candidate_case() -> None:
    broken = _campaign(10.0)
    broken["candidate_anomalies"] = [{}]
    with pytest.raises(GradientArtifactError, match="no case"):
        compare_gradient_campaigns(broken, _campaign(10.0))
