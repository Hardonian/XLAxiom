"""Checksummed, resumable campaign-shard ledger with no executable payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .storage import ArtifactError, atomic_write_text

LEDGER_VERSION = "1.0"


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def load_ledger(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read campaign ledger: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("campaign ledger must be a JSON object")
    checksum = data.get("sha256")
    payload = {key: value for key, value in data.items() if key != "sha256"}
    if payload.get("ledger_version") != LEDGER_VERSION or not isinstance(checksum, str):
        raise ArtifactError("unsupported or malformed campaign ledger")
    if _checksum(payload) != checksum:
        raise ArtifactError("campaign ledger checksum mismatch")
    if not isinstance(payload.get("shards"), dict):
        raise ArtifactError("campaign ledger shards must be an object")
    return data


def save_shard(
    path: str | Path, run_identity: dict[str, Any], shard: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Atomically append/replace one exact shard; mismatched campaigns cannot mix."""
    target = Path(path)
    if target.exists():
        existing = load_ledger(target)
        payload = {key: value for key, value in existing.items() if key != "sha256"}
        if payload["run_identity"] != run_identity:
            raise ArtifactError("ledger belongs to a different campaign identity")
    else:
        payload = {"ledger_version": LEDGER_VERSION, "run_identity": run_identity, "shards": {}}
    shard_key = _canonical(shard)
    payload["shards"][shard_key] = {"shard": shard, "result": result}
    record = {**payload, "sha256": _checksum(payload)}
    atomic_write_text(target, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def completed_shard(
    ledger: dict[str, Any], run_identity: dict[str, Any], shard: dict[str, Any]
) -> dict[str, Any] | None:
    """Return only a shard matching the exact device/budget identity."""
    loaded = load_ledger_data(ledger)
    if loaded["run_identity"] != run_identity:
        return None
    entry = loaded["shards"].get(_canonical(shard))
    if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
        return None
    return entry["result"]


def load_ledger_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-loaded ledger without a filesystem round trip."""
    checksum = data.get("sha256")
    payload = {key: value for key, value in data.items() if key != "sha256"}
    if (
        payload.get("ledger_version") != LEDGER_VERSION
        or not isinstance(checksum, str)
        or _checksum(payload) != checksum
        or not isinstance(payload.get("shards"), dict)
    ):
        raise ArtifactError("invalid campaign ledger")
    return payload
