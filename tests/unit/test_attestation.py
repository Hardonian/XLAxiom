from pathlib import Path

from xlaxiom.attestation import attestation_payload


def test_attestation_payload_binds_only_the_subject_name_and_digest(tmp_path: Path) -> None:
    subject = tmp_path / "evidence.json"
    subject.write_text('{"ok":true}\n', encoding="utf-8")

    payload = attestation_payload(subject)

    assert payload["subject"] == "evidence.json"
    assert len(payload["subject_sha256"]) == 64
    assert "tmp" not in payload["subject"]
