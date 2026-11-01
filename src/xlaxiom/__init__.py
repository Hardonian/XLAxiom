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
from .golden import UPDATE_ENV_VAR, check_golden, load_golden, save_golden
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
from .safetensors_io import (
    INDEX_FILENAME,
    read_checkpoint,
    read_safetensors,
    write_safetensors,
)
from .serving import (
    TopKParityReport,
    audit_topk_parity,
    huggingface_topk,
    openai_compatible_topk,
    topk_from_logits,
)
from .tracing import capture, extract_subgraph, trace_jax
from .triage import format_triage_issue
from .ui import generate_visualizer_html
from .ulp import UlpStats, assert_ulp_close, max_ulp_error, ulp_distance, ulp_stats

__all__ = [
    "CIRegressionGateResult",
    "CalibratedTolerances",
    "CheckpointParityReport",
    "HardwareDriftReport",
    "INDEX_FILENAME",
    "KernelFuzzResult",
    "LayerMetric",
    "LayerParityReport",
    "NumericalAssertionError",
    "NumericalDiff",
    "ParityResult",
    "QuantizationAuditReport",
    "SCHEMA_VERSION",
    "ServingParityReport",
    "TopKParityReport",
    "UPDATE_ENV_VAR",
    "UlpStats",
    "Workload",
    "__version__",
    "assert_close",
    "assert_ulp_close",
    "audit_checkpoint_parity",
    "audit_hardware_drift",
    "audit_layer_parity",
    "audit_quantization_risk",
    "audit_serving_parity",
    "audit_topk_parity",
    "bisect_versions",
    "calibrate_tolerances",
    "capture",
    "check_golden",
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
    "huggingface_topk",
    "load_golden",
    "max_ulp_error",
    "metamorphic",
    "openai_compatible_topk",
    "read_checkpoint",
    "read_safetensors",
    "save_golden",
    "test_invariants",
    "topk_from_logits",
    "trace_jax",
    "ulp_distance",
    "ulp_stats",
    "write_safetensors",
]
