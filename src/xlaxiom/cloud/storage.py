"""Google Cloud Storage (GCS) URI handler for transparent artifact reading and writing.

Enables native reading and writing of checksummed cases, relational artifacts,
campaign ledgers, and attestations directly from/to ``gs://bucket/path`` URIs,
with support for Google Cloud ADC credentials and an in-memory/local mock for
isolated testing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_GCS_URI_PATTERN = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<blob>.+)$")

# In-memory mock storage for testing and offline environments
_GCS_MOCK_STORAGE: dict[str, str] = {}


def is_gcs_uri(path: str | Path) -> bool:
    """Return True if path represents a Google Cloud Storage URI (gs://bucket/blob)."""
    return str(path).startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/blob into (bucket_name, blob_name)."""
    match = _GCS_URI_PATTERN.match(uri)
    if not match:
        raise ValueError(f"invalid GCS URI format: {uri!r} (expected 'gs://<bucket>/<blob>')")
    return match.group("bucket"), match.group("blob")


def read_gcs_text(uri: str) -> str:
    """Read text from a GCS URI using google-cloud-storage if available, else mock."""
    if uri in _GCS_MOCK_STORAGE:
        return _GCS_MOCK_STORAGE[uri]

    try:
        from google.cloud import storage as gcs_storage

        bucket_name, blob_name = parse_gcs_uri(uri)
        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")
    except ImportError:
        # Check if set in mock or raise descriptive error
        if uri in _GCS_MOCK_STORAGE:
            return _GCS_MOCK_STORAGE[uri]
        raise RuntimeError(
            f"cannot read from {uri}: 'google-cloud-storage' package is not installed "
            f"and no mock object is registered"
        ) from None
    except Exception as exc:
        raise OSError(f"failed to read GCS blob {uri}: {exc}") from exc


def write_gcs_text(uri: str, contents: str) -> str:
    """Write text to a GCS URI atomically using google-cloud-storage if available, else mock."""
    bucket_name, blob_name = parse_gcs_uri(uri)

    try:
        from google.cloud import storage as gcs_storage

        client = gcs_storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(contents, content_type="application/json; charset=utf-8")
        return uri
    except ImportError:
        # Store in mock registry
        _GCS_MOCK_STORAGE[uri] = contents
        return uri
    except Exception as exc:
        # If client fails due to credentials in test environment, fallback to mock if requested
        if os.environ.get("XLAXIOM_MOCK_GCS", "0") == "1":
            _GCS_MOCK_STORAGE[uri] = contents
            return uri
        raise OSError(f"failed to write GCS blob {uri}: {exc}") from exc


def clear_mock_gcs() -> None:
    """Clear in-memory GCS mock storage (useful in tests)."""
    _GCS_MOCK_STORAGE.clear()
