"""Detached OpenSSL signatures for release evidence, without storing private keys."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .storage import ArtifactError, atomic_write_text

ATTESTATION_VERSION = "1.0"


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def attestation_payload(subject: str | Path) -> dict[str, Any]:
    """Build the immutable subject statement to be signed by an external key holder."""
    path = Path(subject)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ArtifactError(f"cannot read attestation subject: {exc}") from exc
    return {
        "attestation_version": ATTESTATION_VERSION,
        "subject": path.name,
        "subject_sha256": digest,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "signature_algorithm": "openssl-dgst-sha256",
    }


def sign_attestation(subject: str | Path, private_key: str | Path) -> dict[str, Any]:
    """Produce an OpenSSL detached signature; the private key is never read into Python."""
    payload = attestation_payload(subject)
    try:
        completed = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", str(private_key)],
            input=_canonical(payload),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ArtifactError("OpenSSL is required for signing and was not found") from exc
    if completed.returncode != 0 or not completed.stdout:
        raise ArtifactError(f"OpenSSL signing failed: {completed.stderr.decode('utf-8', 'replace')[-500:]}")
    return {**payload, "signature_base64": base64.b64encode(completed.stdout).decode("ascii")}


def verify_attestation(attestation: dict[str, Any], public_key: str | Path) -> bool:
    """Verify a detached signature and reject missing/future schema fields."""
    signature_text = attestation.get("signature_base64")
    payload = {key: value for key, value in attestation.items() if key != "signature_base64"}
    if payload.get("attestation_version") != ATTESTATION_VERSION or not isinstance(signature_text, str):
        raise ArtifactError("malformed attestation")
    try:
        signature = base64.b64decode(signature_text, validate=True)
    except ValueError as exc:
        raise ArtifactError("malformed base64 signature") from exc
    try:
        with tempfile.TemporaryDirectory(prefix="xlaxiom-attestation-") as directory:
            signature_path = Path(directory) / "signature.bin"
            payload_path = Path(directory) / "payload.json"
            signature_path.write_bytes(signature)
            payload_path.write_bytes(_canonical(payload))
            completed = subprocess.run(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-verify",
                    str(public_key),
                    "-signature",
                    str(signature_path),
                    str(payload_path),
                ],
                capture_output=True,
                check=False,
            )
    except OSError as exc:
        raise ArtifactError("OpenSSL is required for verification and was not found") from exc
    return completed.returncode == 0


def verify_subject_attestation(
    attestation: dict[str, Any], subject: str | Path, public_key: str | Path
) -> bool:
    """Verify both the detached signature and the current subject-file digest."""
    try:
        current_digest = hashlib.sha256(Path(subject).read_bytes()).hexdigest()
    except OSError as exc:
        raise ArtifactError(f"cannot read attestation subject: {exc}") from exc
    if current_digest != attestation.get("subject_sha256"):
        return False
    return verify_attestation(attestation, public_key)


def load_attestation(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read attestation: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("attestation must be a JSON object")
    return data


def save_attestation(path: str | Path, attestation: dict[str, Any]) -> Path:
    if not isinstance(attestation.get("signature_base64"), str):
        raise ArtifactError("attestation has no detached signature")
    return atomic_write_text(path, json.dumps(attestation, indent=2, sort_keys=True) + "\n")
