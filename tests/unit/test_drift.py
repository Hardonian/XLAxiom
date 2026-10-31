"""Unit tests for cross-hardware numerical drift auditing."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.drift import audit_hardware_drift


def test_audit_hardware_drift_exact_match() -> None:
    base = np.ones((10, 10), dtype=np.float32)
    outputs = {
        "cpu": base,
        "cuda-h100": base.copy(),
        "tpu-v5e": base.copy(),
    }
    report = audit_hardware_drift(outputs, baseline_device="cpu")

    assert report.baseline_device == "cpu"
    assert "cuda-h100" in report.drift_signatures
    assert report.drift_signatures["cuda-h100"] == "Bitwise / Exact Match"
    assert report.drift_signatures["tpu-v5e"] == "Bitwise / Exact Match"
    assert "Target Device" in report.format_summary()


def test_audit_hardware_drift_architectural_signatures() -> None:
    base = np.ones((10, 10), dtype=np.float32)
    # Simulate TF32 truncation (rel error ~ 1e-3)
    cuda_tf32 = base + 1e-3
    # Simulate BF16 truncation (rel error ~ 1e-2)
    tpu_bf16 = base + 1e-2
    # Simulate severe divergence
    rogue_device = base * 2.0

    outputs = {
        "cpu": base,
        "cuda-tf32": cuda_tf32,
        "tpu-bf16": tpu_bf16,
        "rogue": rogue_device,
    }
    report = audit_hardware_drift(outputs, baseline_device="cpu", rtol=1e-5, atol=1e-5)

    assert "TF32 Truncation" in report.drift_signatures["cuda-tf32"]
    assert "BFloat16 Truncation" in report.drift_signatures["tpu-bf16"]
    assert "Severe Algorithmic Divergence" in report.drift_signatures["rogue"]

    summary = report.format_summary()
    assert "cuda-tf32" in summary
    assert "DRIFT" in summary


def test_audit_hardware_drift_missing_baseline() -> None:
    outputs = {"cuda": np.ones(5)}
    with pytest.raises(ValueError, match="Baseline device 'cpu' not found"):
        audit_hardware_drift(outputs, baseline_device="cpu")

