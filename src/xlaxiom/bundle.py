"""Safe portable bundle creation and validation using checksummed JSON and ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from .storage import load_case


def create_bundle(cases: list[str], output: str) -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    for item in cases:
        path = Path(item)
        artifact = load_case(path)
        entries.append({"path": f"cases/{path.name}", "sha256": artifact.to_dict()["sha256"]})
    manifest = {"bundle_version": "1.0", "cases": entries, "instructions": "xlaxiom bundle run --bundle bundle.zip"}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for item, entry in zip(cases, entries, strict=True):
            archive.write(item, entry["path"])
    return manifest


def validate_bundle(path: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        if any(name.startswith("/") or ".." in Path(name).parts for name in archive.namelist()):
            raise ValueError("unsafe archive path")
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["cases"]:
            payload = json.loads(archive.read(entry["path"]))
            if payload.get("sha256") != entry["sha256"]:
                raise ValueError(f"checksum mismatch for {entry['path']}")
    return manifest
