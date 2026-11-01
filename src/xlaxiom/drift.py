"""Cross-hardware numerical drift auditing and precision signature classification.

Identifies hardware-specific floating-point discrepancies across CPU, NVIDIA GPU (TF32),
Google Cloud TPU (BF16), and Apple Silicon (MPS), distinguishing compiler bugs from
expected architectural rounding (FMA contraction, 10-bit TF32 mantissa truncation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .assertions import NumericalDiff, _to_numpy, diff


@dataclass(frozen=True)
class HardwareDriftReport:
    """Audit of numerical drift across different accelerator architectures."""

    baseline_device: str
    devices_evaluated: tuple[str, ...]
    per_device_diffs: dict[str, NumericalDiff]
    drift_signatures: dict[str, str]

    def format_summary(self) -> str:
        """Format terminal diagnostic report."""
        header = f"{'Target Device':<18} {'Status':<8} {'Max Abs':<11} {'SNR (dB)':<10} Signature"
        lines = [
            f"=== Cross-Hardware Drift Audit (Baseline: {self.baseline_device}) ===",
            f"Devices evaluated: {', '.join(self.devices_evaluated)}",
            "",
            header,
            "-" * 75,
        ]
        for dev in self.devices_evaluated:
            if dev == self.baseline_device:
                continue
            d = self.per_device_diffs.get(dev)
            if not d:
                continue
            sig = self.drift_signatures.get(dev, "unknown")
            st = "PASS" if d.is_close else "DRIFT"
            snr_str = f"{d.snr_db:.1f}" if not math.isinf(d.snr_db) else "inf"
            lines.append(f"{dev:<18} {st:<8} {d.max_absolute_error:<11.2e} {snr_str:<10} {sig}")
        lines.append("")
        lines.append("Signatures are heuristics inferred from error magnitude, not proof of cause.")
        return "\n".join(lines)


def _classify_drift_signature(d: NumericalDiff) -> str:
    """Classify the root architectural cause of hardware floating-point drift."""
    if d.is_close:
        return "Bitwise / Exact Match"
    if d.shape_a != d.shape_b:
        return "Shape Mismatch"
    if d.nan_mismatches or d.inf_mismatches:
        return "Non-Finite Values (NaN/Inf)"
    # TF32 truncation (10-bit mantissa) typical relative error ~ 1e-3 (2^-10 = ~9.7e-4)
    if 5e-4 <= d.max_relative_error <= 5e-3:
        return "NVIDIA TF32 Truncation (10-bit mantissa)"
    # BF16 truncation (7-bit mantissa) typical relative error ~ 1e-2 (2^-7 = ~7.8e-3)
    if 5e-3 < d.max_relative_error <= 3e-2:
        return "TPU/GPU BFloat16 Truncation (7-bit mantissa)"
    # FMA contraction (very small relative drift, order of 1e-6 in fp32)
    if d.max_relative_error < 5e-4 and d.snr_db > 60.0:
        return "FMA Associative Contraction"
    if d.max_relative_error > 0.1:
        return "Severe Algorithmic Divergence"
    return "Floating-Point Rounding Drift"


def audit_hardware_drift(
    outputs_by_device: dict[str, Any],
    *,
    baseline_device: str = "cpu",
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> HardwareDriftReport:
    """Audit numerical output drift across multiple hardware targets against a baseline.

    Parameters
    ----------
    outputs_by_device : dict[str, Any]
        Dictionary mapping device name (e.g. 'cpu', 'cuda-h100', 'tpu-v5e', 'apple-mps')
        to evaluated output tensors.
    baseline_device : str, optional
        Key of the baseline reference device (default: 'cpu').
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    HardwareDriftReport
        Comprehensive drift report with architectural cause signatures.
    """
    if baseline_device not in outputs_by_device:
        keys = list(outputs_by_device.keys())
        raise ValueError(
            f"Baseline device '{baseline_device}' not found in outputs_by_device keys: {keys}"
        )

    base_tensor = _to_numpy(outputs_by_device[baseline_device])
    devices = tuple(outputs_by_device.keys())

    diffs: dict[str, NumericalDiff] = {}
    signatures: dict[str, str] = {}

    for dev, tensor in outputs_by_device.items():
        if dev == baseline_device:
            continue
        arr = _to_numpy(tensor)
        d = diff(arr, base_tensor, rtol=rtol, atol=atol)
        diffs[dev] = d
        signatures[dev] = _classify_drift_signature(d)

    return HardwareDriftReport(
        baseline_device=baseline_device,
        devices_evaluated=devices,
        per_device_diffs=diffs,
        drift_signatures=signatures,
    )
