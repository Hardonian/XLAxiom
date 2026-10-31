"""Unit tests for HuggingFace checkpoint and serving parity auditing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xlaxiom.hf import (
    _load_weights,
    audit_checkpoint_parity,
    audit_serving_parity,
)


def test_audit_checkpoint_parity_exact_match() -> None:
    weights = {
        "encoder.layer.0.weight": np.ones((16, 16), dtype=np.float32),
        "encoder.layer.0.bias": np.zeros(16, dtype=np.float32),
    }
    report = audit_checkpoint_parity(weights, weights)

    assert report.is_identical is True
    assert report.total_weights == 2
    assert report.matched_weights == 2
    assert report.regressed_weights == 0
    assert report.missing_keys == ()
    assert report.extra_keys == ()
    assert "Checkpoint Weight Parity: PASSED" in report.format_summary()


def test_audit_checkpoint_parity_divergent_and_missing() -> None:
    weights_a = {
        "layer1.weight": np.ones((8, 8), dtype=np.float32),
        "layer2.weight": np.ones((8, 8), dtype=np.float32),
    }
    weights_b = {
        "layer1.weight": np.ones((8, 8), dtype=np.float32) * 5.0,  # divergent
        "layer3.weight": np.zeros((8, 8), dtype=np.float32),  # extra
    }
    report = audit_checkpoint_parity(weights_a, weights_b, rtol=1e-4, atol=1e-4)

    assert report.is_identical is False
    assert report.total_weights == 1
    assert report.regressed_weights == 1
    assert report.missing_keys == ("layer2.weight",)
    assert report.extra_keys == ("layer3.weight",)

    summary = report.format_summary()
    assert "WEIGHT DIVERGENCE DETECTED" in summary
    assert "Missing keys in candidate" in summary
    assert "Extra keys in candidate" in summary


def test_load_weights_npz(tmp_path: Path) -> None:
    npz_file = tmp_path / "model.npz"
    w1 = np.arange(10, dtype=np.float32)
    w2 = np.ones((4, 4), dtype=np.float32)
    np.savez(npz_file, w1=w1, w2=w2)

    loaded = _load_weights(npz_file)
    assert "w1" in loaded and "w2" in loaded
    np.testing.assert_array_equal(loaded["w1"], w1)
    np.testing.assert_array_equal(loaded["w2"], w2)


def test_load_weights_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        _load_weights("non_existent_file.npz")


def test_audit_serving_parity_identical() -> None:
    def model_fn(x: np.ndarray) -> np.ndarray:
        return x @ np.eye(x.shape[-1], dtype=np.float32)

    inputs = [np.random.default_rng(42).standard_normal((1, 8), dtype=np.float32) for _ in range(5)]
    report = audit_serving_parity(model_fn, model_fn, inputs)

    assert report.is_aligned is True
    assert report.top1_agreement_pct == 100.0
    assert report.top5_overlap_pct == 100.0
    assert report.max_logit_difference == 0.0
    assert report.mean_kl_divergence < 1e-9
    assert "Serving Engine Parity: PASSED" in report.format_summary()


def test_audit_serving_parity_divergence() -> None:
    rng = np.random.default_rng(42)

    def engine_fn(x: np.ndarray) -> np.ndarray:
        return x + 5.0

    def ref_fn(x: np.ndarray) -> np.ndarray:
        return x * -1.0

    inputs = [rng.standard_normal((1, 10), dtype=np.float32) for _ in range(3)]
    report = audit_serving_parity(engine_fn, ref_fn, inputs, kl_tolerance=1e-5)

    assert report.is_aligned is False
    assert report.mean_kl_divergence > 0.0
    assert "LOGIT DIVERGENCE DETECTED" in report.format_summary()
