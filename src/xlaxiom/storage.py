"""Checksummed portable JSON evidence artifacts; no pickle or executable payloads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .comparison import ComparisonResult
from .model import Workload
from .pipeline import FaultInjection

CASE_VERSION = "1.0"
RELATIONAL_CASE_VERSION = "1.0"


class ArtifactError(ValueError):
    """Artifact is malformed or its checksum does not match."""


def atomic_write_text(path: str | Path, contents: str) -> Path:
    """Write complete UTF-8 text beside its destination, then atomically replace it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


@dataclass(frozen=True)
class CaseArtifact:
    workload: Workload
    expected_classification: str | None = None
    comparison_mode: str = "jax-jit"
    fault: FaultInjection | None = None
    provenance: str = "generated"
    minimization: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "case_version": CASE_VERSION,
            "workload": self.workload.canonical_dict(),
            "expected_classification": self.expected_classification,
            "comparison_mode": self.comparison_mode,
            "fault": asdict(self.fault) if self.fault else None,
            "provenance": self.provenance,
            "minimization": self.minimization,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseArtifact:
        checksum = data.get("sha256")
        payload = {key: value for key, value in data.items() if key != "sha256"}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if not isinstance(checksum, str) or not hashlib.sha256(encoded).hexdigest() == checksum:
            raise ArtifactError("artifact checksum mismatch")
        if payload.get("case_version") != CASE_VERSION:
            raise ArtifactError("unsupported case version")
        fault_data = payload.get("fault")
        try:
            return cls(
                workload=Workload.from_dict(payload["workload"]),
                expected_classification=payload.get("expected_classification"),
                comparison_mode=str(payload.get("comparison_mode", "jax-jit")),
                fault=FaultInjection(**fault_data) if fault_data else None,
                provenance=str(payload.get("provenance", "generated")),
                minimization=payload.get("minimization"),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ArtifactError(f"malformed case artifact: {exc}") from exc


def save_case(path: str | Path, artifact: CaseArtifact) -> Path:
    return atomic_write_text(path, json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n")


def load_case(path: str | Path) -> CaseArtifact:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read case artifact: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("case artifact must be a JSON object")
    return CaseArtifact.from_dict(data)


@dataclass(frozen=True)
class RelationalArtifact:
    """A checksummed pair of declarative workloads with an output relation.

    The pair is intentionally data-only.  It supports minimized metamorphic failures
    without accepting a callback, expression, or executable payload from an artifact.
    """

    identifier: str
    left: Workload
    right: Workload
    expected_classification: str | None = None
    backend: str = "jax-jit"
    relation: str = "outputs_equal"
    provenance: str = "metamorphic-property"
    minimization: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        if self.left.policy != self.right.policy:
            raise ArtifactError("relational workloads must use the same comparison policy")
        return {
            "relational_case_version": RELATIONAL_CASE_VERSION,
            "identifier": self.identifier,
            "left": self.left.canonical_dict(),
            "right": self.right.canonical_dict(),
            "expected_classification": self.expected_classification,
            "backend": self.backend,
            "relation": self.relation,
            "provenance": self.provenance,
            "minimization": self.minimization,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationalArtifact:
        checksum = data.get("sha256")
        payload = {key: value for key, value in data.items() if key != "sha256"}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if not isinstance(checksum, str) or hashlib.sha256(encoded).hexdigest() != checksum:
            raise ArtifactError("artifact checksum mismatch")
        if payload.get("relational_case_version") != RELATIONAL_CASE_VERSION:
            raise ArtifactError("unsupported relational case version")
        try:
            artifact = cls(
                identifier=str(payload["identifier"]),
                left=Workload.from_dict(payload["left"]),
                right=Workload.from_dict(payload["right"]),
                expected_classification=payload.get("expected_classification"),
                backend=str(payload.get("backend", "jax-jit")),
                relation=str(payload.get("relation", "outputs_equal")),
                provenance=str(payload.get("provenance", "metamorphic-property")),
                minimization=payload.get("minimization"),
            )
            artifact.payload()
            if not artifact.identifier or artifact.relation != "outputs_equal":
                raise ArtifactError("unsupported relational artifact fields")
            return artifact
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"malformed relational artifact: {exc}") from exc


Artifact = CaseArtifact | RelationalArtifact


def save_relational_case(path: str | Path, artifact: RelationalArtifact) -> Path:
    return atomic_write_text(path, json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n")


def load_relational_case(path: str | Path) -> RelationalArtifact:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read relational case artifact: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("relational case artifact must be a JSON object")
    return RelationalArtifact.from_dict(data)


def artifact_from_dict(data: dict[str, Any]) -> Artifact:
    """Load either portable artifact type without guessing from executable fields."""
    if "relational_case_version" in data:
        return RelationalArtifact.from_dict(data)
    return CaseArtifact.from_dict(data)


def load_artifact(path: str | Path) -> Artifact:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read artifact: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("artifact must be a JSON object")
    return artifact_from_dict(data)


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    return atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def observation_record(artifact: CaseArtifact, comparison: ComparisonResult) -> dict[str, Any]:
    return {
        "case_sha256": artifact.to_dict()["sha256"],
        "provenance": artifact.provenance,
        "synthetic_fault": asdict(artifact.fault) if artifact.fault else None,
        "comparison": comparison.to_dict(),
    }
