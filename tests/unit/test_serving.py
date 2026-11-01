"""Unit tests for top-k serving parity and adapters."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.serving import (
    audit_topk_parity,
    topk_from_logits,
)


def test_topk_parity_aligned() -> None:
    def model_a(prompt: str) -> dict[str, float]:
        return {"Paris": -0.1, "London": -2.3, "Berlin": -4.5}

    def model_b(prompt: str) -> dict[str, float]:
        return {"Paris": -0.1002, "London": -2.299, "Berlin": -4.51}

    report = audit_topk_parity(model_b, model_a, ["Capital of France?"])
    assert report.is_aligned is True
    assert report.top1_agreement_pct == 100.0
    assert report.topk_overlap_pct == 100.0
    assert report.mean_abs_logprob_diff < 0.01
    assert "PASSED" in report.format_summary()


def test_topk_parity_diverged() -> None:
    def model_a(prompt: str) -> dict[str, float]:
        return {"apple": -0.2, "banana": -1.5}

    def model_b(prompt: str) -> dict[str, float]:
        return {"orange": -0.1, "grape": -2.0}

    report = audit_topk_parity(
        model_b, model_a, ["A fruit:"], min_top1_agreement_pct=90.0, min_overlap_pct=50.0
    )
    assert report.is_aligned is False
    assert report.top1_agreement_pct == 0.0
    assert report.topk_overlap_pct == 0.0
    assert "TOP-K DIVERGENCE DETECTED" in report.format_summary()


def test_topk_from_logits_adapter() -> None:
    vocab = ["zero", "one", "two", "three"]

    def dummy_logits(prompt: str) -> np.ndarray:
        # Give token 'two' (index 2) the highest logit
        return np.array([1.0, 2.0, 10.0, 0.5])

    adapter = topk_from_logits(dummy_logits, lambda i: vocab[i], k=2)
    result = adapter("test prompt")

    assert "two" in result
    assert "one" in result
    assert result["two"] > result["one"]


def test_topk_empty_prompts_raises() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        audit_topk_parity(lambda p: {"a": -0.1}, lambda p: {"a": -0.1}, [])


def test_topk_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        audit_topk_parity(lambda p: {"a": -0.1}, lambda p: {"a": -0.1}, ["prompt"], k=0)


def test_topk_nonfinite_logprob_raises() -> None:
    with pytest.raises(ValueError, match="Non-finite logprob"):
        audit_topk_parity(lambda p: {"a": float("nan")}, lambda p: {"a": -0.1}, ["prompt"])
