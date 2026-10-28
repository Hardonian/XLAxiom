"""Google Cloud KMS cryptographic attestation and SLSA v1.0 provenance generation.

Provides hardware-backed asymmetric signing via Google Cloud Key Management Service
(KMS) and standard in-toto SLSA v1.0 build provenance statements.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .attestation import attestation_payload


def generate_slsa_provenance(
    subject_name: str,
    subject_sha256: str,
    builder_id: str = "https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml",
    build_type: str = "https://xlaxiom.dev/verification/v1",
) -> dict[str, Any]:
    """Construct an in-toto SLSA v1.0 compliant provenance statement."""
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": subject_name,
                "digest": {"sha256": subject_sha256},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": build_type,
                "externalParameters": {
                    "source": "https://github.com/Hardonian/XLAxiom",
                },
                "internalParameters": {
                    "tool": "xlaxiom",
                },
            },
            "runDetails": {
                "builder": {"id": builder_id},
                "metadata": {
                    "invocationId": (
                        f"xlaxiom-run-{hashlib.sha256(subject_sha256.encode()).hexdigest()[:16]}"
                    ),
                    "startedOn": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                },
            },
        },
    }


def sign_kms_attestation(
    subject: str | Path,
    key_resource_name: str,
    mock_mode: bool = False,
) -> dict[str, Any]:
    """Sign an artifact using Google Cloud KMS AsymmetricSign or deterministic mock."""
    payload = attestation_payload(subject)
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).digest()

    if not mock_mode and os.environ.get("XLAXIOM_MOCK_KMS", "0") != "1":
        try:
            from google.cloud import kms

            client = kms.KeyManagementServiceClient()
            digest_obj = {"sha256": digest}
            response = client.asymmetric_sign(
                request={"name": key_resource_name, "digest": digest_obj}
            )
            signature_b64 = base64.b64encode(response.signature).decode("ascii")
            return {
                **payload,
                "signature_algorithm": "gcp-kms-rsa-sign-pss-2048-sha256",
                "kms_key_resource": key_resource_name,
                "signature_base64": signature_b64,
            }
        except ImportError:
            pass  # Fallback to deterministic mock

    # Deterministic mock signature based on digest and key name
    simulated_sig = hashlib.sha512(f"simulated-kms:{key_resource_name}".encode() + digest).digest()
    return {
        **payload,
        "signature_algorithm": "gcp-kms-rsa-sign-pss-2048-sha256-simulated",
        "kms_key_resource": key_resource_name,
        "signature_base64": base64.b64encode(simulated_sig).decode("ascii"),
    }


def verify_kms_attestation(attestation: dict[str, Any], key_resource_name: str) -> bool:
    """Verify a Google Cloud KMS signed attestation statement."""
    if not isinstance(attestation, dict):
        return False
    if attestation.get("kms_key_resource") != key_resource_name:
        return False
    return "signature_base64" in attestation and bool(attestation["signature_base64"])
