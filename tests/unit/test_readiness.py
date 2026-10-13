from xlaxiom.comparison import Classification
from xlaxiom.generation import synthetic_control
from xlaxiom.readiness import assess_upstream_readiness
from xlaxiom.storage import CaseArtifact


def test_readiness_never_assumes_missing_external_reproduction() -> None:
    artifact = CaseArtifact(
        synthetic_control(),
        expected_classification=Classification.NUMERICAL_DISCREPANCY.value,
        minimization={"accepted_reductions": 1},
    )

    assessment = assess_upstream_readiness(artifact)

    assert assessment["ready"] is False
    assert not next(
        check for check in assessment["checks"] if check["name"] == "fresh_process_reproduction"
    )["passed"]
