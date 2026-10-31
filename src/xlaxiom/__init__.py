"""XLAxiom: evidence-first differential testing, kernel fuzzing, and numerical verification."""

from ._version import __version__
from .assertions import (
    NumericalAssertionError,
    NumericalDiff,
    ParityResult,
    assert_close,
    check_parity,
    diff,
)
from .bisect import bisect_versions
from .execution import execute_array_api, execute_torch
from .fuzzing import KernelFuzzResult, fuzz_kernel
from .hardware import discover_apple_silicon, discover_rocm_devices
from .metamorphic_decorator import test_invariants
from .model import SCHEMA_VERSION, Workload
from .nn_parity import LayerMetric, LayerParityReport, audit_layer_parity
from .quantization import QuantizationAuditReport, audit_quantization_risk
from .tracing import capture, extract_subgraph, trace_jax
from .triage import format_triage_issue

__all__ = [
    "SCHEMA_VERSION",
    "KernelFuzzResult",
    "LayerMetric",
    "LayerParityReport",
    "NumericalAssertionError",
    "NumericalDiff",
    "ParityResult",
    "QuantizationAuditReport",
    "Workload",
    "__version__",
    "assert_close",
    "audit_layer_parity",
    "audit_quantization_risk",
    "bisect_versions",
    "capture",
    "check_parity",
    "diff",
    "discover_apple_silicon",
    "discover_rocm_devices",
    "execute_array_api",
    "execute_torch",
    "extract_subgraph",
    "format_triage_issue",
    "fuzz_kernel",
    "test_invariants",
    "trace_jax",
]
