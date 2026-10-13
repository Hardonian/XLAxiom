import hashlib
import json
import zipfile

import pytest

from xlaxiom.bundle import create_bundle, validate_bundle
from xlaxiom.generation import synthetic_control
from xlaxiom.metamorphic import build_property_artifact
from xlaxiom.storage import ArtifactError, CaseArtifact, save_case, save_relational_case


def test_portable_bundle_round_trip(tmp_path) -> None:
    case = save_case(tmp_path / "case.json", CaseArtifact(synthetic_control()))
    bundle = tmp_path / "bundle.zip"
    create_bundle([str(case)], str(bundle))
    assert validate_bundle(str(bundle))["cases"][0]["path"] == "cases/case.json"


def test_bundle_output_is_reproducible(tmp_path) -> None:
    case = save_case(tmp_path / "case.json", CaseArtifact(synthetic_control()))
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    create_bundle([str(case)], str(first))
    create_bundle([str(case)], str(second))
    assert first.read_bytes() == second.read_bytes()


def test_bundle_rejects_duplicate_member_names(tmp_path) -> None:
    first = tmp_path / "first" / "case.json"
    second = tmp_path / "second" / "case.json"
    first.parent.mkdir()
    second.parent.mkdir()
    save_case(first, CaseArtifact(synthetic_control(1)))
    save_case(second, CaseArtifact(synthetic_control(2)))
    with pytest.raises(ArtifactError, match="duplicate case name"):
        create_bundle([str(first), str(second)], str(tmp_path / "bundle.zip"))


def test_bundle_validation_rejects_non_case_payload(tmp_path) -> None:
    bundle = tmp_path / "invalid.zip"
    payload = {"case_version": "0.0"}
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "bundle_version": "1.0",
        "cases": [{"path": "cases/not-a-case.json", "sha256": checksum}],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("cases/not-a-case.json", json.dumps({**payload, "sha256": checksum}))
    with pytest.raises(ArtifactError, match="unsupported case version"):
        validate_bundle(str(bundle))


def test_bundle_validation_rejects_undeclared_members(tmp_path) -> None:
    case = save_case(tmp_path / "case.json", CaseArtifact(synthetic_control()))
    bundle = tmp_path / "bundle.zip"
    create_bundle([str(case)], str(bundle))
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("notes.txt", "not an evidence artifact")
    with pytest.raises(ArtifactError, match="undeclared archive members"):
        validate_bundle(str(bundle))


def test_bundle_accepts_portable_relational_artifacts(tmp_path) -> None:
    relation = save_relational_case(
        tmp_path / "relation.json",
        build_property_artifact("add-commutativity-v1", seed=2, backend="numpy"),
    )
    bundle = tmp_path / "relation.zip"

    create_bundle([str(relation)], str(bundle))

    assert validate_bundle(str(bundle))["cases"][0]["path"] == "cases/relation.json"
