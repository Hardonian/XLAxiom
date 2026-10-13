"""Safe portable bundle creation and validation using checksummed JSON and ZIP."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .pipeline import run_comparison
from .storage import ArtifactError, load_case, save_json

BUNDLE_VERSION = "1.0"
MAX_UNCOMPRESSED_SIZE = 20_000_000


def _archive_info(path: str) -> zipfile.ZipInfo:
    """Create reproducible, non-executable ZIP member metadata."""
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def create_bundle(cases: list[str], output: str) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    payloads: list[tuple[str, bytes]] = []
    if not cases:
        raise ArtifactError("bundle requires at least one case")
    for item in cases:
        path = Path(item)
        artifact = load_case(path)
        member_path = f"cases/{path.name}"
        if any(entry["path"] == member_path for entry in entries):
            raise ArtifactError(f"bundle contains duplicate case name {path.name!r}")
        entries.append({"path": member_path, "sha256": artifact.to_dict()["sha256"]})
        payloads.append((member_path, path.read_bytes()))
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "cases": entries,
        "instructions": "xlaxiom bundle run --bundle bundle.zip",
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            _archive_info("manifest.json"),
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        for member_path, payload in payloads:
            archive.writestr(_archive_info(member_path), payload)
    return manifest


def validate_bundle(path: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(
                name.startswith("/") or ".." in Path(name).parts for name in names
            ):
                raise ArtifactError("unsafe archive path")
            if sum(item.file_size for item in archive.infolist()) > MAX_UNCOMPRESSED_SIZE:
                raise ArtifactError("bundle exceeds uncompressed size limit")
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (KeyError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"invalid bundle manifest: {exc}") from exc
            if not isinstance(manifest, dict) or manifest.get("bundle_version") != BUNDLE_VERSION:
                raise ArtifactError("unsupported bundle version")
            entries = manifest.get("cases")
            if not isinstance(entries, list) or not entries:
                raise ArtifactError("bundle requires a non-empty cases list")
            member_paths: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ArtifactError("bundle case entry must be an object")
                member_path = entry.get("path")
                checksum = entry.get("sha256")
                if (
                    not isinstance(member_path, str)
                    or not member_path.startswith("cases/")
                    or member_path not in names
                    or member_path in member_paths
                    or not isinstance(checksum, str)
                ):
                    raise ArtifactError("invalid bundle case entry")
                member_paths.add(member_path)
                try:
                    payload = json.loads(archive.read(member_path))
                except json.JSONDecodeError as exc:
                    raise ArtifactError(f"invalid JSON case in {member_path}") from exc
                if not isinstance(payload, dict) or payload.get("sha256") != checksum:
                    raise ArtifactError(f"checksum mismatch for {member_path}")
                # A valid bundle is executable evidence, not merely a ZIP with matching strings.
                load_case_from_payload(payload)
            if set(names) != {"manifest.json", *member_paths}:
                raise ArtifactError("bundle contains undeclared archive members")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArtifactError(f"cannot read bundle: {exc}") from exc
    return manifest


def run_bundle(bundle: str, output: str) -> dict[str, Any]:
    """Execute declared cases directly from validated archive bytes; no extraction is needed."""
    manifest = validate_bundle(bundle)
    results: list[dict[str, Any]] = []
    with zipfile.ZipFile(bundle) as archive:
        for entry in manifest["cases"]:
            payload = json.loads(archive.read(entry["path"]))
            artifact = load_case_from_payload(payload)
            comparison = run_comparison(artifact.workload, artifact.comparison_mode, artifact.fault)
            results.append({"case_sha256": entry["sha256"], "comparison": comparison.to_dict()})
    evidence = {"evidence_version": "1.0", "bundle_manifest": manifest, "results": results}
    save_json(output, evidence)
    return evidence


def load_case_from_payload(payload: dict[str, Any]):
    from .storage import CaseArtifact

    return CaseArtifact.from_dict(payload)
