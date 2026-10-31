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
from .calibration import CalibratedTolerances, calibrate_tolerances
from .ci import CIRegressionGateResult, evaluate_ci_gate, generate_pr_comment
from .drift import HardwareDriftReport, audit_hardware_drift
from .execution import execute_array_api, execute_torch
from .fuzzing import KernelFuzzResult, fuzz_kernel
from .hardware import discover_apple_silicon, discover_rocm_devices
from .hf import (
    CheckpointParityReport,
    ServingParityReport,
    audit_checkpoint_parity,
    audit_serving_parity,
)
from .metamorphic_decorator import metamorphic, test_invariants
from .model import SCHEMA_VERSION, Workload
from .nn_parity import LayerMetric, LayerParityReport, audit_layer_parity
from .quantization import QuantizationAuditReport, audit_quantization_risk
from .tracing import capture, extract_subgraph, trace_jax
from .triage import format_triage_issue
from .ui import generate_visualizer_html

__all__ = [
    "CIRegressionGateResult",
    "CalibratedTolerances",
    "CheckpointParityReport",
    "HardwareDriftReport",
    "KernelFuzzResult",
    "LayerMetric",
    "LayerParityReport",
    "NumericalAssertionError",
    "NumericalDiff",
    "ParityResult",
    "QuantizationAuditReport",
    "SCHEMA_VERSION",
    "ServingParityReport",
    "Workload",
    "__version__",
    "assert_close",
    "audit_checkpoint_parity",
    "audit_hardware_drift",
    "audit_layer_parity",
    "audit_quantization_risk",
    "audit_serving_parity",
    "bisect_versions",
    "calibrate_tolerances",
    "capture",
    "check_parity",
    "diff",
    "discover_apple_silicon",
    "discover_rocm_devices",
    "evaluate_ci_gate",
    "execute_array_api",
    "execute_torch",
    "extract_subgraph",
    "format_triage_issue",
    "fuzz_kernel",
    "generate_pr_comment",
    "generate_visualizer_html",
    "metamorphic",
    "test_invariants",
    "trace_jax",
]
