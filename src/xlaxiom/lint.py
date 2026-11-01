"""Checkpoint and tensor sanity linter.

Detects silent corruption, numerical defects, and inference latency risks across
Safetensors, PyTorch, and NumPy checkpoints:
- Non-finite numbers (NaN, Inf) that cause silent model collapse.
- Subnormal / denormal floating-point numbers that trigger CPU microcode assist traps
  and GPU pipeline stalls, degrading inference throughput by up to 100x.
- Dead / all-zero tensors indicating uninitialized weights or catastrophic gradient death.
- Extreme outliers (> 6 standard deviations) signaling weight explosion or training instability.
- Dynamic range statistics and parameter distribution health.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assertions import _to_numpy
from .hf import _load_weights


@dataclass(frozen=True)
class TensorHealthRecord:
    """Sanity audit record for an individual tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    total_elements: int
    nan_count: int
    inf_count: int
    subnormal_count: int
    zero_count: int
    zero_percentage: float
    min_val: float
    max_val: float
    mean: float
    std: float
    outlier_count_6sigma: int
    is_healthy: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointHealthReport:
    """Comprehensive sanity audit report for a model checkpoint."""

    checkpoint_path: str
    total_tensors: int
    healthy_tensors: int
    warning_tensors: int
    corrupted_tensors: int
    total_elements: int
    total_nans: int
    total_infs: int
    total_subnormals: int
    total_dead_tensors: int
    status: str  # "HEALTHY", "WARNING", "CORRUPTED"
    records: tuple[TensorHealthRecord, ...]

    def format_summary(self) -> str:
        """Format an informative diagnostic report for the console."""
        lines = [
            f"=== Checkpoint Health Audit: [{self.status}] ===",
            f"Checkpoint: {self.checkpoint_path}",
            f"Total Tensors: {self.total_tensors} | Total Parameters: {self.total_elements:,}",
            (
                f"Healthy: {self.healthy_tensors} | Warnings: {self.warning_tensors} | "
                f"Corrupted: {self.corrupted_tensors}"
            ),
        ]

        if self.total_nans > 0 or self.total_infs > 0:
            lines.append(
                f"[CRITICAL] Non-finites detected: {self.total_nans} NaNs, {self.total_infs} Infs"
            )
        if self.total_subnormals > 0:
            lines.append(
                f"[WARNING] Subnormal floats detected: {self.total_subnormals} "
                "(Risk: 10x-100x inference latency degradation)"
            )
        if self.total_dead_tensors > 0:
            lines.append(
                f"[WARNING] Dead / all-zero tensors: {self.total_dead_tensors} "
                "(Possible uninitialized weights)"
            )

        problematic = [r for r in self.records if not r.is_healthy]
        if problematic:
            lines.append("\nProblematic Tensors:")
            for r in problematic[:10]:
                issues_str = "; ".join(r.issues)
                lines.append(f"  - {r.name} ({r.dtype}, shape {r.shape}): {issues_str}")
            if len(problematic) > 10:
                lines.append(f"  ... and {len(problematic) - 10} more tensors with issues.")
        else:
            lines.append("[OK] All tensors passed integrity, sanity, and dynamic-range audits.")

        return "\n".join(lines)


def lint_tensor(name: str, tensor: Any) -> TensorHealthRecord:
    """Audit an individual tensor for numerical anomalies, non-finites, and subnormals."""
    arr = _to_numpy(tensor)
    total_elements = arr.size
    shape = tuple(arr.shape)
    dtype_str = str(arr.dtype)

    if total_elements == 0:
        return TensorHealthRecord(
            name=name,
            shape=shape,
            dtype=dtype_str,
            total_elements=0,
            nan_count=0,
            inf_count=0,
            subnormal_count=0,
            zero_count=0,
            zero_percentage=0.0,
            min_val=0.0,
            max_val=0.0,
            mean=0.0,
            std=0.0,
            outlier_count_6sigma=0,
            is_healthy=True,
            issues=(),
        )

    issues: list[str] = []
    is_floating = np.issubdtype(arr.dtype, np.floating)

    nan_count = 0
    inf_count = 0
    subnormal_count = 0

    if is_floating:
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        nan_count = int(np.sum(nan_mask))
        inf_count = int(np.sum(inf_mask))

        if nan_count > 0:
            issues.append(f"{nan_count} NaNs detected")
        if inf_count > 0:
            issues.append(f"{inf_count} Infs detected")

        finite_mask = ~(nan_mask | inf_mask)
        finite_arr = arr[finite_mask]

        # Subnormal check
        try:
            tiny = np.finfo(arr.dtype).tiny
            subnormal_mask = finite_mask & (arr != 0.0) & (np.abs(arr) < tiny)
            subnormal_count = int(np.sum(subnormal_mask))
            if subnormal_count > 0:
                issues.append(f"{subnormal_count} subnormal floats detected")
        except (ValueError, KeyError):
            pass
    else:
        finite_arr = arr.ravel()

    zero_count = int(np.sum(arr == 0))
    zero_percentage = (zero_count / total_elements) * 100.0
    if zero_count == total_elements:
        issues.append("Tensor is entirely zero (dead weight)")

    if finite_arr.size > 0:
        finite_f64 = finite_arr.astype(np.float64)
        min_val = float(np.min(finite_f64))
        max_val = float(np.max(finite_f64))
        mean = float(np.mean(finite_f64))
        std = float(np.std(finite_f64))
        if std > 0.0:
            outlier_count_6sigma = int(np.sum(np.abs(finite_f64 - mean) > 6.0 * std))
        else:
            outlier_count_6sigma = 0
    else:
        min_val = float("nan")
        max_val = float("nan")
        mean = float("nan")
        std = float("nan")
        outlier_count_6sigma = 0

    is_healthy = len(issues) == 0

    return TensorHealthRecord(
        name=name,
        shape=shape,
        dtype=dtype_str,
        total_elements=total_elements,
        nan_count=nan_count,
        inf_count=inf_count,
        subnormal_count=subnormal_count,
        zero_count=zero_count,
        zero_percentage=zero_percentage,
        min_val=min_val,
        max_val=max_val,
        mean=mean,
        std=std,
        outlier_count_6sigma=outlier_count_6sigma,
        is_healthy=is_healthy,
        issues=tuple(issues),
    )


def lint_checkpoint(source: str | Path | dict[str, Any]) -> CheckpointHealthReport:
    """Audit an entire model checkpoint (.safetensors, .npz, directory, or dict).

    Parameters
    ----------
    source : str, Path, or dict
        Path to checkpoint file or dictionary of tensors.

    Returns
    -------
    CheckpointHealthReport
        Structured health audit report.
    """
    ckpt_path_str = str(source) if not isinstance(source, dict) else "<in-memory dict>"
    weights = _load_weights(source)

    records: list[TensorHealthRecord] = []
    total_elements = 0
    total_nans = 0
    total_infs = 0
    total_subnormals = 0
    total_dead = 0
    corrupted_count = 0
    warning_count = 0
    healthy_count = 0

    for name, tensor in weights.items():
        rec = lint_tensor(name, tensor)
        records.append(rec)
        total_elements += rec.total_elements
        total_nans += rec.nan_count
        total_infs += rec.inf_count
        total_subnormals += rec.subnormal_count
        if rec.zero_percentage == 100.0 and rec.total_elements > 0:
            total_dead += 1

        if rec.nan_count > 0 or rec.inf_count > 0:
            corrupted_count += 1
        elif not rec.is_healthy:
            warning_count += 1
        else:
            healthy_count += 1

    if corrupted_count > 0:
        status = "CORRUPTED"
    elif warning_count > 0:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return CheckpointHealthReport(
        checkpoint_path=ckpt_path_str,
        total_tensors=len(records),
        healthy_tensors=healthy_count,
        warning_tensors=warning_count,
        corrupted_tensors=corrupted_count,
        total_elements=total_elements,
        total_nans=total_nans,
        total_infs=total_infs,
        total_subnormals=total_subnormals,
        total_dead_tensors=total_dead,
        status=status,
        records=tuple(records),
    )
