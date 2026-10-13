"""Checksummed host-environment evidence for independent reproduction review."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .storage import ArtifactError, atomic_write_text

ENVIRONMENT_EVIDENCE_VERSION = "1.0"


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def capture_environment(label: str) -> dict[str, Any]:
    """Capture non-secret, portable runtime identity for an evidence attachment."""
    if not label.strip():
        raise ArtifactError("environment evidence label is required")
    jax_info: dict[str, Any] = {"available": False, "devices": [], "x64_enabled": None}
    try:
        import jax as jax_module
        import jaxlib

        jax_info = {
            "available": True,
            "version": jax_module.__version__,
            "jaxlib": jaxlib.__version__,
            "devices": [str(device) for device in jax_module.devices()],
            "x64_enabled": bool(getattr(jax_module.config, "jax_enable_x64", False)),
        }
    except (ImportError, RuntimeError):
        pass
    payload = {
        "environment_evidence_version": ENVIRONMENT_EVIDENCE_VERSION,
        "label": label,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "jax": jax_info,
        "redaction": (
            "No host name, user name, paths, GPU UUIDs, or environment variables are stored."
        ),
    }
    return {**payload, "sha256": _digest(payload)}


def validate_environment_evidence(data: dict[str, Any]) -> dict[str, Any]:
    """Validate checksum and reject a future schema rather than guessing its meaning."""
    checksum = data.get("sha256")
    payload = {key: value for key, value in data.items() if key != "sha256"}
    if payload.get("environment_evidence_version") != ENVIRONMENT_EVIDENCE_VERSION:
        raise ArtifactError("unsupported environment evidence version")
    if not isinstance(checksum, str) or _digest(payload) != checksum:
        raise ArtifactError("environment evidence checksum mismatch")
    if not isinstance(payload.get("label"), str) or not isinstance(payload.get("host"), dict):
        raise ArtifactError("malformed environment evidence")
    return data


def save_environment(path: str | Path, evidence: dict[str, Any]) -> Path:
    validate_environment_evidence(evidence)
    return atomic_write_text(path, json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def load_environment(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read environment evidence: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("environment evidence must be a JSON object")
    return validate_environment_evidence(data)


def compare_environments(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Determine whether two retained records establish a meaningfully independent host."""
    validate_environment_evidence(primary)
    validate_environment_evidence(secondary)
    primary_host = primary["host"]
    secondary_host = secondary["host"]
    changed = [
        name
        for name in ("system", "release", "machine", "python", "executable")
        if primary_host.get(name) != secondary_host.get(name)
    ]
    distinct_labels = primary["label"] != secondary["label"]
    independent = bool(changed) and distinct_labels and primary["sha256"] != secondary["sha256"]
    return {
        "assessment_version": "1.0",
        "independent": independent,
        "primary_sha256": primary["sha256"],
        "secondary_sha256": secondary["sha256"],
        "host_differences": changed,
        "distinct_labels": distinct_labels,
        "reason": (
            "records have distinct labels and independently varying host identity"
            if independent
            else (
                "records do not establish an independently varying host; "
                "attach a second host record"
            )
        ),
    }
