"""Unit tests for quantization and precision risk auditing."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.quantization import audit_quantization_risk


def test_audit_quantization_well_behaved_tensor() -> None:
    # Standard normal tensor with moderate values
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 100))

    report = audit_quantization_risk(x, target_dtype="fp8_e4m3")
    assert report.total_elements == 10000
    assert report.risk_level in ("LOW", "MODERATE")
    assert report.snr_db > 15.0
    assert "FP8 (E4M3)" in report.format_summary()

    html = report._repr_html_()
    assert "Quantization Audit: FP8 (E4M3)" in html


def test_audit_quantization_severe_outliers_critical_risk() -> None:
    # Tensor with extreme outliers that cause severe clipping
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1000,))
    # Inject 5% extreme outliers
    x[:50] = 50000.0

    report = audit_quantization_risk(x, target_dtype="fp4_e2m1", percentile=90.0)
    assert report.clipped_elements > 0
    assert report.clipping_percentage > 0.0
    assert report.risk_level == "CRITICAL"
    assert "Recommendation" in report.format_summary()


def test_audit_quantization_empty_tensor() -> None:
    x = np.empty((0, 5))
    report = audit_quantization_risk(x, target_dtype="bfloat16")
    assert report.total_elements == 0
    assert report.risk_level == "LOW"


def test_audit_quantization_invalid_dtype() -> None:
    x = np.ones((4, 4))
    with pytest.raises(ValueError, match="Unsupported target_dtype"):
        audit_quantization_risk(x, target_dtype="fp2_unknown")
