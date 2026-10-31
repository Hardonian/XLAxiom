"""Unit tests for metamorphic invariant test decorators."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.assertions import NumericalAssertionError
from xlaxiom.metamorphic_decorator import test_invariants as xl_invariants


def test_invariant_batch_padding_success() -> None:
    # A standard elementwise or reduction operation that preserves batch independence
    @xl_invariants(invariants=["batch_padding"], sample_shape=(4, 16))
    def linear_layer(x: np.ndarray) -> np.ndarray:
        return x * 2.0 + 1.0

    # Running with no args uses generated synthetic sample
    linear_layer()

    # Running with custom args
    custom_x = np.ones((8, 16), dtype=np.float32)
    out = linear_layer(custom_x)
    assert out.shape == (8, 16)


def test_invariant_scale_invariance_success() -> None:
    # Softmax is scale invariant: softmax(c * x) == softmax(x) for c=1 or normalized
    # Let's test a simple normalization layer: x / norm(x)
    @xl_invariants(invariants=["scale_invariance"])
    def normalize_layer(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return x / norm

    normalize_layer()


def test_invariant_violation_raises() -> None:
    # An operation that is NOT scale invariant (e.g. adding a constant)
    @xl_invariants(invariants=["scale_invariance"])
    def add_constant(x: np.ndarray) -> np.ndarray:
        return x + 5.0

    with pytest.raises(NumericalAssertionError, match="Scale invariance failed"):
        add_constant()


def test_unknown_invariant_raises() -> None:
    @xl_invariants(invariants=["nonexistent_invariant"])
    def dummy(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError, match="Unknown invariant"):
        dummy()
