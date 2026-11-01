"""Unit tests for checkpoint and tensor sanity linter."""

from __future__ import annotations

import numpy as np

from xlaxiom.lint import lint_checkpoint, lint_tensor
from xlaxiom.safetensors_io import write_safetensors


def test_lint_tensor_healthy() -> None:
    arr = np.random.randn(10, 10).astype(np.float32)
    record = lint_tensor("layer.weight", arr)
    assert record.is_healthy is True
    assert record.nan_count == 0
    assert record.inf_count == 0
    assert record.subnormal_count == 0
    assert len(record.issues) == 0


def test_lint_tensor_nan_and_inf() -> None:
    arr = np.array([1.0, float("nan"), 2.0, float("inf")], dtype=np.float32)
    record = lint_tensor("broken.layer", arr)
    assert record.is_healthy is False
    assert record.nan_count == 1
    assert record.inf_count == 1
    assert any("NaNs detected" in issue for issue in record.issues)
    assert any("Infs detected" in issue for issue in record.issues)


def test_lint_tensor_subnormal() -> None:
    # Subnormal for float32 is non-zero with absolute value < 1.175494e-38
    subnormal_val = np.float32(1e-40)
    arr = np.array([1.0, subnormal_val, 2.0], dtype=np.float32)
    record = lint_tensor("subnormal.layer", arr)
    assert record.is_healthy is False
    assert record.subnormal_count == 1
    assert any("subnormal floats detected" in issue for issue in record.issues)


def test_lint_tensor_dead_zero() -> None:
    arr = np.zeros((5, 5), dtype=np.float32)
    record = lint_tensor("dead.bias", arr)
    assert record.is_healthy is False
    assert record.zero_percentage == 100.0
    assert any("entirely zero" in issue for issue in record.issues)


def test_lint_checkpoint_dict() -> None:
    weights = {
        "encoder.weight": np.ones((4, 4), dtype=np.float32),
        "encoder.bias": np.zeros((4,), dtype=np.float32),  # dead weight
        "decoder.weight": np.array([1.0, float("nan")], dtype=np.float32),  # corrupted
    }
    report = lint_checkpoint(weights)
    assert report.total_tensors == 3
    assert report.healthy_tensors == 1
    assert report.warning_tensors == 1
    assert report.corrupted_tensors == 1
    assert report.status == "CORRUPTED"
    summary = report.format_summary()
    assert "[CRITICAL] Non-finites detected" in summary
    assert "decoder.weight" in summary


def test_lint_checkpoint_safetensors(tmp_path) -> None:
    ckpt_path = tmp_path / "model.safetensors"
    weights = {
        "w1": np.ones((8, 8), dtype=np.float32),
        "w2": np.ones((8,), dtype=np.float32) * 2.0,
    }
    write_safetensors(ckpt_path, weights)

    report = lint_checkpoint(ckpt_path)
    assert report.status == "HEALTHY"
    assert report.total_tensors == 2
    assert report.healthy_tensors == 2
    assert report.total_nans == 0
    assert report.total_subnormals == 0
    assert "[OK] All tensors passed integrity" in report.format_summary()
