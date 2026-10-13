from xlaxiom.environment import (
    capture_environment,
    compare_environments,
    validate_environment_evidence,
)


def test_environment_evidence_is_checksummed_and_fail_closed_for_same_host() -> None:
    first = capture_environment("host-a")
    second = capture_environment("host-b")

    validate_environment_evidence(first)
    assessment = compare_environments(first, second)

    assert assessment["independent"] is False
    assert "second host" in assessment["reason"]
