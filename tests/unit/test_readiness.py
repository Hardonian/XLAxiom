import pytest

from xlaxiom.comparison import Classification, ComparisonResult
from xlaxiom.environment import capture_environment
from xlaxiom.generation import synthetic_control
from xlaxiom.model import ComparisonPolicy
from xlaxiom.readiness import (
    ReadinessEvidenceError,
    assess_upstream_readiness,
    build_reproduction_evidence,
)
from xlaxiom.storage import CaseArtifact


def _candidate_artifact() -> CaseArtifact:
    return CaseArtifact(
        synthetic_control(),
        expected_classification=Classification.NUMERICAL_DISCREPANCY.value,
        minimization={"accepted_reductions": 1},
    )


def _discrepant_comparison() -> ComparisonResult:
    return ComparisonResult(
        classification=Classification.NUMERICAL_DISCREPANCY,
        explanation="test discrepancy",
        policy=ComparisonPolicy(),
    )


def test_readiness_never_assumes_missing_external_reproduction() -> None:
    artifact = _candidate_artifact()

    assessment = assess_upstream_readiness(artifact)

    assert assessment["ready"] is False
    assert not next(
        check for check in assessment["checks"] if check["name"] == "fresh_process_reproduction"
    )["passed"]


def test_readiness_accepts_case_bound_reproductions_from_independent_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _candidate_artifact()
    primary_environment = capture_environment("host-a")
    monkeypatch.setattr("xlaxiom.environment.platform.system", lambda: "IndependentOS")
    secondary_environment = capture_environment("host-b")

    primary_reproduction = build_reproduction_evidence(
        artifact, _discrepant_comparison(), primary_environment
    )
    secondary_reproduction = build_reproduction_evidence(
        artifact, _discrepant_comparison(), secondary_environment
    )

    assessment = assess_upstream_readiness(
        artifact,
        (primary_reproduction, secondary_reproduction),
        (primary_environment, secondary_environment),
    )

    assert assessment["ready"] is True
    assert all(check["passed"] for check in assessment["checks"])
    assert assessment["evidence"]["independent_reproduction_pairs"]


def test_readiness_rejects_reproduction_evidence_for_another_case() -> None:
    artifact = _candidate_artifact()
    other_artifact = CaseArtifact(
        synthetic_control(seed=9),
        expected_classification=Classification.NUMERICAL_DISCREPANCY.value,
        minimization={"accepted_reductions": 1},
    )
    reproduction = build_reproduction_evidence(other_artifact, _discrepant_comparison())

    with pytest.raises(ReadinessEvidenceError, match="does not match the supplied case"):
        assess_upstream_readiness(artifact, (reproduction,))


def test_readiness_requires_every_bound_environment_record() -> None:
    artifact = _candidate_artifact()
    environment = capture_environment("host-a")
    reproduction = build_reproduction_evidence(artifact, _discrepant_comparison(), environment)

    with pytest.raises(ReadinessEvidenceError, match="was not supplied"):
        assess_upstream_readiness(artifact, (reproduction,))
