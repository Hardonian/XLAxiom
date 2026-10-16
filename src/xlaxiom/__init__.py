"""XLAxiom: evidence-first differential testing for numerical compiler workloads."""

from ._version import __version__
from .bisect import bisect_versions
from .execution import execute_array_api, execute_torch
from .hardware import discover_apple_silicon, discover_rocm_devices
from .model import SCHEMA_VERSION, Workload
from .tracing import capture, extract_subgraph, trace_jax
from .triage import format_triage_issue

__all__ = [
    "SCHEMA_VERSION",
    "Workload",
    "__version__",
    "bisect_versions",
    "capture",
    "discover_apple_silicon",
    "discover_rocm_devices",
    "execute_array_api",
    "execute_torch",
    "extract_subgraph",
    "format_triage_issue",
    "trace_jax",
]
