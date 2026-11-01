"""Regression tests for defects found during the post-3.0.0 QA review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xlaxiom.assertions import diff
from xlaxiom.calibration import calibrate_tolerances
from xlaxiom.ci import evaluate_ci_gate, generate_pr_comment
from xlaxiom.drift import audit_hardware_drift
from xlaxiom.hf import audit_checkpoint_parity, audit_serving_parity
from xlaxiom.ui import generate_visualizer_html

# --------------------------------------------------------------------------- ci


def test_pr_comment_is_ascii_only() -> None:
    """Printing the comment must not crash legacy cp1252 Windows consoles."""
    good = diff(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    bad = diff(np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32))
    comment = generate_pr_comment([("ok", good), ("bad", bad)])
    comment.encode("ascii")


def test_pr_comment_agrees_with_snr_gate() -> None:
    """A tensor inside tolerance but below the SNR floor must fail in the comment too."""
    ref = np.linspace(1.0, 2.0, 64).astype(np.float32)
    noisy = ref + np.float32(1e-3)
    d = diff(noisy, ref, rtol=1e-1, atol=1e-1)
    assert d.is_close
    assert d.snr_db < 80.0

    gate = evaluate_ci_gate([("t", d)], min_snr_db=80.0)
    assert gate.passed is False
    assert gate.regressed_tensors == 1
    assert "REGRESSION DETECTED" in gate.pr_comment_markdown
    assert "[! 1]" in gate.pr_comment_markdown


def test_pr_comment_badge_url_has_no_raw_space() -> None:
    bad = diff(np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    badge = generate_pr_comment(bad).splitlines()[0]
    assert " " not in badge.split("(", 1)[1]


def test_pr_comment_escapes_table_pipes() -> None:
    d = diff(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))
    comment = generate_pr_comment([("a|b", d)])
    assert "`a\\|b`" in comment


def test_gate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_ci_gate([])


# --------------------------------------------------------------------------- hf


def test_checkpoint_summary_lists_worst_weights_first() -> None:
    base = {f"w{i}": np.ones(8, dtype=np.float32) for i in range(8)}
    cand = {k: v.copy() for k, v in base.items()}
    cand["w7"] = np.full(8, 50.0, dtype=np.float32)  # last key, most divergent
    report = audit_checkpoint_parity(base, cand)
    assert "w7" in report.format_summary()


def test_npz_checkpoint_handle_released(tmp_path: Path) -> None:
    path = tmp_path / "w.npz"
    np.savez(path, a=np.ones(3, dtype=np.float32))
    report = audit_checkpoint_parity(path, {"a": np.ones(3, dtype=np.float32)})
    assert report.is_identical
    path.unlink()  # raises PermissionError on Windows if the handle leaked


def test_uppercase_suffix_supported(tmp_path: Path) -> None:
    path = tmp_path / "W.NPZ"
    np.savez(tmp_path / "w.npz", a=np.ones(3, dtype=np.float32))
    (tmp_path / "w.npz").rename(path)
    assert audit_checkpoint_parity(path, path).is_identical


def _logits(seq: int = 6, vocab: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(seq, vocab)).astype(np.float32)


def test_serving_percentages_bounded_for_sequence_logits() -> None:
    """Per-position counts previously divided by prompt count, yielding values > 100%."""
    x = _logits()
    report = audit_serving_parity(lambda _: x, lambda _: x, [0, 1])
    assert report.top1_agreement_pct == pytest.approx(100.0)
    assert report.top5_overlap_pct == pytest.approx(100.0)
    assert report.is_aligned


def test_serving_nan_logits_never_align() -> None:
    ref = _logits()
    bad = ref.copy()
    bad[0, 0] = np.nan
    report = audit_serving_parity(lambda _: bad, lambda _: ref, [0])
    assert report.is_aligned is False
    assert report.non_finite_prompts == 1
    assert "NaN" in report.format_summary()


def test_serving_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        audit_serving_parity(lambda x: x, lambda x: x, [])


def test_serving_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        audit_serving_parity(lambda _: _logits(4), lambda _: _logits(5), [0])


def test_serving_masked_neg_inf_logits_are_valid() -> None:
    ref = _logits()
    ref[:, :3] = -np.inf
    report = audit_serving_parity(lambda _: ref, lambda _: ref, [0])
    assert report.is_aligned
    assert report.mean_kl_divergence == pytest.approx(0.0, abs=1e-12)


def test_serving_rtol_atol_are_enforced_when_given() -> None:
    ref = _logits()
    shifted = ref + np.float32(5e-3)  # same softmax -> KL ~ 0, but logits moved
    loose = audit_serving_parity(lambda _: shifted, lambda _: ref, [0])
    assert loose.is_aligned and loose.logits_within_tolerance is None
    tight = audit_serving_parity(lambda _: shifted, lambda _: ref, [0], rtol=0.0, atol=1e-4)
    assert tight.is_aligned is False
    assert tight.logits_within_tolerance is False


def test_serving_kl_detects_tail_probability_shift() -> None:
    ref = np.zeros((1, 8), dtype=np.float64)
    eng = ref.copy()
    eng[0, 0] = 5.0
    report = audit_serving_parity(lambda _: eng, lambda _: ref, [0], min_top1_agreement_pct=0.0)
    assert report.mean_kl_divergence > 1e-4
    assert report.is_aligned is False


# ------------------------------------------------------------------ calibration


def test_calibration_rejects_nan_output() -> None:
    with pytest.raises(ValueError, match="NaN/inf"):
        calibrate_tolerances(lambda x: x * np.nan, np.ones(3), runs=2)


def test_calibration_rejects_shape_change() -> None:
    sizes = iter([3, 4])
    with pytest.raises(ValueError, match="shape"):
        calibrate_tolerances(lambda: np.ones(next(sizes)), runs=2)


def test_calibration_rejects_sub_unity_safety_factor() -> None:
    with pytest.raises(ValueError, match="safety_factor"):
        calibrate_tolerances(lambda x: x, np.ones(3), runs=2, safety_factor=0.5)


def test_calibration_summary_reports_actual_safety_factor() -> None:
    cal = calibrate_tolerances(lambda x: x, np.ones(3), runs=2, safety_factor=4.0)
    assert "4x safety headroom" in cal.format_summary()


# ------------------------------------------------------------------------ drift


def test_drift_flags_shape_mismatch_and_nan() -> None:
    base = np.ones((2, 2), dtype=np.float32)
    report = audit_hardware_drift(
        {
            "cpu": base,
            "wrong-shape": np.ones((3, 2), dtype=np.float32),
            "nan": np.full((2, 2), np.nan, dtype=np.float32),
        }
    )
    assert report.drift_signatures["wrong-shape"] == "Shape Mismatch"
    assert "Non-Finite" in report.drift_signatures["nan"]
    assert "heuristics" in report.format_summary()


# --------------------------------------------------------------------------- ui


def test_visualizer_survives_non_finite_errors() -> None:
    d = diff(np.array([np.nan, 1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32))
    doc = generate_visualizer_html(d)
    assert "<svg" in doc


def test_visualizer_is_responsive_and_escapes_title() -> None:
    d = diff(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))
    doc = generate_visualizer_html(d, title="<script>alert(1)</script>")
    assert 'name="viewport"' in doc
    assert "<script>alert(1)</script>" not in doc

