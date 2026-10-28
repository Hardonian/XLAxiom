"""Google Cloud Logging and OpenTelemetry distributed tracing integration.

Provides structured Cloud Logging payloads and OpenTelemetry compiler execution spans
for tracking XLA JIT compilation overhead versus kernel execution latency.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any


def format_cloud_log(
    message: str,
    severity: str = "INFO",
    component: str = "xlaxiom-compiler-audit",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format a structured JSON payload conforming to Google Cloud Logging."""
    return {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "severity": severity.upper(),
        "message": message,
        "logging.googleapis.com/sourceLocation": {
            "component": component,
        },
        "labels": {
            "tool": "xlaxiom",
            "version": "2.1.0",
        },
        "jsonPayload": metadata or {},
    }


@contextmanager
def trace_compiler_span(
    operation_name: str,
    workload_digest: str,
    backend: str = "jax-jit",
) -> Generator[dict[str, Any], None, None]:
    """Context manager tracing compiler compilation and execution durations."""
    span: dict[str, Any] = {
        "operation": operation_name,
        "workload_digest": workload_digest,
        "backend": backend,
        "start_time_iso": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "compilation_ms": 0.0,
        "execution_ms": 0.0,
        "total_ms": 0.0,
    }
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        span["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
