"""Safe portable bundle creation and validation using checksummed JSON and ZIP."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .storage import load_case
from .pipeline import run_comparison
from .storage import save_json


def create_bundle(cases: list[str], output: str) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    for item in cases:
        path = Path(item)
        artifact = load_case(path)
        entries.append({"path": f"cases/{path.name}", "sha256": artifact.to_dict()["sha256"]})
    manifest = {
        "bundle_version": "1.0",
        "cases": entries,
        "instructions": "xlaxiom bundle run --bundle bundle.zip",
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for item, entry in zip(cases, entries, strict=True):
            archive.write(item, entry["path"])
    return manifest


def validate_bundle(path: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith("/") or ".." in Path(name).parts for name in names
        ):
            raise ValueError("unsafe archive path")
        if sum(item.file_size for item in archive.infolist()) > 20_000_000:
            raise ValueError("bundle exceeds uncompressed size limit")
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["cases"]:
            payload = json.loads(archive.read(entry["path"]))
            if payload.get("sha256") != entry["sha256"]:
                raise ValueError(f"checksum mismatch for {entry['path']}")
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
