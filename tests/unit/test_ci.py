"""Unit tests for continuous integration regression gate and PR comment bot."""

from __future__ import annotations

import numpy as np

from xlaxiom.assertions import diff
from xlaxiom.ci import evaluate_ci_gate, generate_pr_comment


def test_generate_pr_comment_single_passing() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = diff(arr, arr)
    comment = generate_pr_comment(d, commit_sha="abcdef123456", pr_number=42)

    assert "Numerical%20Parity-PASSED" in comment
    assert "PR #42" in comment
    assert "abcdef1" in comment
    assert "[PASS]" in comment
    assert "Detailed Tensor Parity Breakdown" in comment


def test_generate_pr_comment_with_regression() -> None:
    actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    desired = np.array([1.0, 2.0, 10.0], dtype=np.float32)
    d_fail = diff(actual, desired, rtol=1e-5, atol=1e-5)
    d_pass = diff(actual, actual)

    items = [("layer1.weight", d_pass), ("layer2.weight", d_fail)]
    comment = generate_pr_comment(items, baseline_ref="v2.0.0")

    assert "REGRESSION DETECTED" in comment
    assert "v2.0.0" in comment
    assert "[! 1]" in comment
    assert "layer2.weight" in comment
    assert "Reproduction Code for Regressions" in comment
    assert "assert_close(actual, desired" in comment


def test_evaluate_ci_gate_passing() -> None:
    a = np.ones((10, 10), dtype=np.float32)
    b = a + 1e-7
    d = diff(a, b, rtol=1e-4, atol=1e-4)

    gate = evaluate_ci_gate([("tensor_a", d)], min_snr_db=30.0)
    assert gate.passed is True
    assert gate.regressed_tensors == 0
    assert gate.total_tensors == 1
    assert gate.worst_snr_db > 30.0


def test_evaluate_ci_gate_failing_due_to_snr() -> None:
    a = np.ones((10, 10), dtype=np.float32)
    b = a + 0.05
    d = diff(a, b, rtol=0.1, atol=0.1)  # is_close is True with high rtol

    # But SNR is low (~26 dB), requiring min_snr_db=50.0 should fail gate
    gate = evaluate_ci_gate([("tensor_low_snr", d)], min_snr_db=50.0)
    assert gate.passed is False
    assert gate.regressed_tensors == 1
    assert gate.total_tensors == 1
