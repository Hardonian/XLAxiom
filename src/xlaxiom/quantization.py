"""Quantization and precision-loss risk auditing.

Simulates floating-point truncation across modern AI precision formats:
FP8 (E4M3, E5M2), FP4 (E2M1 / MXFP4), Int8, and BFloat16.
Measures dynamic range headroom, outlier clipping rates, underflow rates,
SNR degradation, and generates actionable precision recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assertions import _to_numpy, diff
from .hf import _load_weights


@dataclass(frozen=True)
class DtypeSpec:
    name: str
    max_val: float
    min_normal: float
    min_subnormal: float
    mantissa_bits: int


DTYPE_SPECS: dict[str, DtypeSpec] = {
    "fp8_e4m3": DtypeSpec(
        name="FP8 (E4M3)",
        max_val=448.0,
        min_normal=2**-6,  # 0.015625
        min_subnormal=2**-9,  # ~0.001953
        mantissa_bits=3,
    ),
    "fp8_e5m2": DtypeSpec(
        name="FP8 (E5M2)",
        max_val=57344.0,
        min_normal=2**-14,  # ~6.10e-5
        min_subnormal=2**-16,  # ~1.52e-5
        mantissa_bits=2,
    ),
    "fp4_e2m1": DtypeSpec(
        name="FP4 (E2M1)",
        max_val=6.0,
        min_normal=1.0,
        min_subnormal=0.5,
        mantissa_bits=1,
    ),
    "int8": DtypeSpec(
        name="INT8",
        max_val=127.0,
        min_normal=1.0,
        min_subnormal=1.0,
        mantissa_bits=0,
    ),
    "bfloat16": DtypeSpec(
        name="BFloat16",
        max_val=3.38e38,
        min_normal=1.17e-38,
        min_subnormal=9.18e-41,
        mantissa_bits=7,
    ),
}


@dataclass(frozen=True)
class QuantizationAuditReport:
    """Audit report for precision loss, dynamic range, and clipping risk."""

    target_dtype: str
    total_elements: int
    optimal_scale: float
    clipped_elements: int
    clipping_percentage: float
    underflow_elements: int
    underflow_percentage: float
    max_absolute_error: float
    snr_db: float
    cosine_similarity: float
    risk_level: str  # "LOW", "MODERATE", "CRITICAL"
    recommendation: str

    def format_summary(self) -> str:
        """Format an informative diagnostic report for console."""
        lines = [
            f"=== Quantization Precision Audit: {self.target_dtype} [{self.risk_level} RISK] ===",
            f"Total elements analyzed: {self.total_elements}",
            f"Calculated scale factor: {self.optimal_scale:.4e}",
            f"Clipped outliers: {self.clipped_elements} ({self.clipping_percentage:.2f}%)",
            f"Flushed underflows: {self.underflow_elements} ({self.underflow_percentage:.2f}%)",
            f"Max absolute error: {self.max_absolute_error:.3e}",
            (
                f"Signal-to-Noise Ratio (SNR): {self.snr_db:.1f} dB | "
                f"Cosine Similarity: {self.cosine_similarity:.5f}"
            ),
            f"Recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Render interactive HTML card for Jupyter/Colab."""
        badge_colors = {
            "LOW": "#10b981",
            "MODERATE": "#f59e0b",
            "CRITICAL": "#ef4444",
        }
        color = badge_colors.get(self.risk_level, "#64748b")
        card_style = (
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 10px 0; "
            "background: #ffffff;"
        )
        badge_style = (
            f"background: {color}; color: white; padding: 3px 10px; border-radius: 9999px; "
            "font-weight: 700; font-size: 11px;"
        )

        return f"""
        <div style="{card_style}">
          <div style="display: flex; align-items: center; justify-content: space-between;
                      margin-bottom: 8px;">
            <div style="font-weight: 700; color: #0f172a; font-size: 15px;">
              Quantization Audit: {self.target_dtype}
            </div>
            <span style="{badge_style}">{self.risk_level} RISK</span>
          </div>
          <p style="margin: 4px 0 10px 0; font-size: 13px; color: #334155;">
            <strong>Recommendation:</strong> {self.recommendation}
          </p>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
                      font-size: 13px;">
            <div style="background: #f8fafc; padding: 8px; border-radius: 6px;">
              <div style="color: #64748b; font-size: 11px;">SNR</div>
              <div style="font-weight: 600; color: #0f172a;">{self.snr_db:.1f} dB</div>
            </div>
            <div style="background: #f8fafc; padding: 8px; border-radius: 6px;">
              <div style="color: #64748b; font-size: 11px;">Outliers Clipped</div>
              <div style="font-weight: 600; color: #0f172a;">{self.clipping_percentage:.2f}%</div>
            </div>
            <div style="background: #f8fafc; padding: 8px; border-radius: 6px;">
              <div style="color: #64748b; font-size: 11px;">Underflow Flushed</div>
              <div style="font-weight: 600; color: #0f172a;">{self.underflow_percentage:.2f}%</div>
            </div>
          </div>
        </div>
        """


def _simulate_quantization(
    arr: np.ndarray[Any, Any], spec: DtypeSpec, scale: float
) -> tuple[np.ndarray[Any, Any], int, int]:
    """Simulate quantization to a target dtype spec using scale factor."""
    scaled = arr / scale

    # Clipping outliers
    clipped_mask = np.abs(scaled) > spec.max_val
    clipped_count = int(np.sum(clipped_mask))
    clipped = np.clip(scaled, -spec.max_val, spec.max_val)

    # Underflow flush to zero
    underflow_mask = (np.abs(clipped) > 0.0) & (np.abs(clipped) < spec.min_subnormal)
    underflow_count = int(np.sum(underflow_mask))
    clipped[underflow_mask] = 0.0

    # Truncation noise simulation based on mantissa bits
    if spec.mantissa_bits > 0:
        # Quantize mantissa steps
        step = 2.0 ** (-spec.mantissa_bits)
        quantized = np.round(clipped / step) * step
    else:
        quantized = np.round(clipped)

    # Re-scale back to original scale
    reconstructed = quantized * scale
    return reconstructed, clipped_count, underflow_count


def audit_quantization_risk(
    tensor_or_model: Any,
    sample_inputs: Any = None,
    *,
    target_dtype: str = "fp8_e4m3",
    scale: float | None = None,
    percentile: float = 99.9,
) -> QuantizationAuditReport:
    """Audit the numerical precision risk of quantizing a tensor to a lower precision format.

    Parameters
    ----------
    tensor_or_model : Tensor, array, or Callable
        The activation tensor or model to audit.
    sample_inputs : Any, optional
        Inputs to pass if `tensor_or_model` is a callable model.
    target_dtype : str, optional
        Target precision format: "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", or "bfloat16".
    scale : float, optional
        Custom scale factor. If None, computes optimal dynamic range scaling based on percentile.
    percentile : float, optional
        Percentile cutoff for scale calculation (default: 99.9).

    Returns
    -------
    QuantizationAuditReport
        Comprehensive audit detailing SNR, clipping rate, underflow rate, and risk recommendations.
    """
    dtype_key = target_dtype.lower().replace("-", "_")
    if dtype_key not in DTYPE_SPECS:
        raise ValueError(
            f"Unsupported target_dtype '{target_dtype}'. Choose from {list(DTYPE_SPECS.keys())}"
        )
    spec = DTYPE_SPECS[dtype_key]

    if callable(tensor_or_model) and sample_inputs is not None:
        inputs = sample_inputs if isinstance(sample_inputs, tuple) else (sample_inputs,)
        raw_output = tensor_or_model(*inputs)
        arr = _to_numpy(raw_output)
    else:
        arr = _to_numpy(tensor_or_model)

    arr = arr.astype(np.float64)
    total_elements = arr.size
    if total_elements == 0:
        return QuantizationAuditReport(
            target_dtype=spec.name,
            total_elements=0,
            optimal_scale=1.0,
            clipped_elements=0,
            clipping_percentage=0.0,
            underflow_elements=0,
            underflow_percentage=0.0,
            max_absolute_error=0.0,
            snr_db=float("inf"),
            cosine_similarity=1.0,
            risk_level="LOW",
            recommendation="Empty array; safe.",
        )

    # Calculate optimal scale if not provided
    if scale is None:
        p_val = float(np.percentile(np.abs(arr), percentile))
        computed_scale = max(1e-12, p_val / spec.max_val)
    else:
        computed_scale = scale

    reconstructed, clipped_count, underflow_count = _simulate_quantization(
        arr, spec, computed_scale
    )

    clip_pct = (clipped_count / total_elements) * 100.0
    underflow_pct = (underflow_count / total_elements) * 100.0

    diff_res = diff(reconstructed, arr, rtol=1e-2, atol=1e-3)
    max_abs_err = diff_res.max_absolute_error
    snr_db = diff_res.snr_db
    cos_sim = diff_res.cosine_similarity

    # Risk assessment
    if clip_pct > 2.0 or underflow_pct > 30.0 or snr_db < 15.0 or cos_sim < 0.99:
        risk = "CRITICAL"
        recommendation = (
            f"High precision degradation: {clip_pct:.1f}% values clipped, "
            f"{underflow_pct:.1f}% underflowed, SNR {snr_db:.1f} dB. "
            f"Do NOT quantize this layer directly; consider SmoothQuant or retain in BF16/FP16."
        )
    elif clip_pct > 0.2 or underflow_pct > 5.0 or snr_db < 25.0 or cos_sim < 0.998:
        risk = "MODERATE"
        recommendation = (
            f"Moderate precision loss: {clip_pct:.2f}% clipped, {underflow_pct:.1f}% underflowed, "
            f"SNR {snr_db:.1f} dB. Safe if coupled with per-channel/per-token dynamic scaling."
        )
    else:
        risk = "LOW"
        recommendation = (
            f"Excellent fidelity: SNR {snr_db:.1f} dB, {clip_pct:.2f}% clipped. "
            f"Safe to deploy in {spec.name}."
        )

    return QuantizationAuditReport(
        target_dtype=spec.name,
        total_elements=total_elements,
        optimal_scale=computed_scale,
        clipped_elements=clipped_count,
        clipping_percentage=clip_pct,
        underflow_elements=underflow_count,
        underflow_percentage=underflow_pct,
        max_absolute_error=max_abs_err,
        snr_db=snr_db,
        cosine_similarity=cos_sim,
        risk_level=risk,
        recommendation=recommendation,
    )


@dataclass(frozen=True)
class CheckpointQuantAuditReport:
    """Comprehensive quantization risk audit across an entire model checkpoint."""

    checkpoint_path: str
    target_dtype: str
    total_weights: int
    low_risk_weights: int
    moderate_risk_weights: int
    critical_risk_weights: int
    average_snr_db: float
    recommended_fallback_layers: tuple[str, ...]
    layer_audits: tuple[tuple[str, QuantizationAuditReport], ...]

    def format_summary(self) -> str:
        """Format an informative diagnostic report for the console."""
        overall_status = (
            "CRITICAL RISK"
            if self.critical_risk_weights > 0
            else ("MODERATE RISK" if self.moderate_risk_weights > 0 else "LOW RISK")
        )
        avg_snr = f"{self.average_snr_db:.1f} dB" if not math.isinf(self.average_snr_db) else "∞"
        lines = [
            f"=== Checkpoint Quantization Audit: {self.target_dtype} [{overall_status}] ===",
            f"Checkpoint: {self.checkpoint_path}",
            f"Total Weights Evaluated: {self.total_weights}",
            (
                f"Risk Breakdown: {self.low_risk_weights} Low, "
                f"{self.moderate_risk_weights} Moderate, {self.critical_risk_weights} Critical"
            ),
            f"Average Simulated SNR: {avg_snr}",
        ]

        if self.critical_risk_weights > 0:
            lines.append(
                f"\n[ACTION REQUIRED] {self.critical_risk_weights} layers have severe "
                f"precision collapse under {self.target_dtype}."
            )
            lines.append("Recommended BF16/FP16 Fallback Whitelist (Do NOT quantize these):")
            for name in self.recommended_fallback_layers[:10]:
                lines.append(f"  - {name}")
            if len(self.recommended_fallback_layers) > 10:
                more = len(self.recommended_fallback_layers) - 10
                lines.append(f"  ... and {more} more layers.")
        else:
            lines.append(
                f"\n[OK] Model is safe for uniform {self.target_dtype} quantization "
                "with standard per-tensor/per-channel scaling."
            )

        return "\n".join(lines)


def audit_checkpoint_quantization(
    source: str | Path | dict[str, Any],
    *,
    target_dtype: str = "fp8_e4m3",
    scale: float | None = None,
    percentile: float = 99.9,
) -> CheckpointQuantAuditReport:
    """Audit quantization risk across all weight tensors in a model checkpoint.

    Parameters
    ----------
    source : str, Path, or dict
        Path to checkpoint file (.safetensors, .npz, directory) or in-memory dictionary.
    target_dtype : str, optional
        Target precision format ('fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'int8', 'bfloat16').
    scale : float, optional
        Explicit scale factor override.
    percentile : float, optional
        Percentile cutoff for scaling (default: 99.9).

    Returns
    -------
    CheckpointQuantAuditReport
        Detailed audit report identifying high-risk layers and mixed-precision recommendations.
    """
    ckpt_path_str = str(source) if not isinstance(source, dict) else "<in-memory dict>"
    weights = _load_weights(source)

    layer_audits: list[tuple[str, QuantizationAuditReport]] = []
    low_count = 0
    mod_count = 0
    crit_count = 0
    fallback_layers: list[str] = []
    snr_list: list[float] = []

    for name, tensor in weights.items():
        audit = audit_quantization_risk(
            tensor, target_dtype=target_dtype, scale=scale, percentile=percentile
        )
        layer_audits.append((name, audit))

        if audit.risk_level == "CRITICAL":
            crit_count += 1
            fallback_layers.append(name)
        elif audit.risk_level == "MODERATE":
            mod_count += 1
        else:
            low_count += 1

        if not math.isinf(audit.snr_db) and not math.isnan(audit.snr_db):
            snr_list.append(audit.snr_db)

    avg_snr = float(np.mean(snr_list)) if snr_list else float("inf")

    return CheckpointQuantAuditReport(
        checkpoint_path=ckpt_path_str,
        target_dtype=target_dtype,
        total_weights=len(layer_audits),
        low_risk_weights=low_count,
        moderate_risk_weights=mod_count,
        critical_risk_weights=crit_count,
        average_snr_db=avg_snr,
        recommended_fallback_layers=tuple(fallback_layers),
        layer_audits=tuple(layer_audits),
    )
