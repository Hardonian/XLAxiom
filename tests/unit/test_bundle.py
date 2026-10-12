from xlaxiom.bundle import create_bundle, validate_bundle
from xlaxiom.generation import synthetic_control
from xlaxiom.storage import CaseArtifact, save_case


def test_portable_bundle_round_trip(tmp_path) -> None:
    case = save_case(tmp_path / "case.json", CaseArtifact(synthetic_control()))
    bundle = tmp_path / "bundle.zip"
    create_bundle([str(case)], str(bundle))
    assert validate_bundle(str(bundle))["cases"][0]["path"] == "cases/case.json"
