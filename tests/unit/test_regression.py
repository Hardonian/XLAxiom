from xlaxiom.regression import VersionClassification, classify_versions


def test_version_classification_transitions() -> None:
    passed = {"worker_status": "ok", "comparison": {"classification": "pass"}}
    failed = {"worker_status": "ok", "comparison": {"classification": "numerical_discrepancy"}}
    assert classify_versions(passed, passed)["classification"] == VersionClassification.STABLE_PASS
    assert classify_versions(failed, passed)["classification"] == VersionClassification.FIXED
    assert (
        classify_versions(passed, failed)["classification"] == VersionClassification.NEW_REGRESSION
    )
