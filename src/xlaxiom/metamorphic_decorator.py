"""Hypothesis-like metamorphic invariant test decorators for neural networks and tensor operations.

Verifies mathematical invariants such as batch-padding invariance, scale equivariance/invariance,
shift invariance (e.g. Softmax), and precision stability across input perturbations.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .assertions import _to_numpy, assert_close


def _verify_batch_padding(
    fn: Callable[..., Any],
    sample_input: np.ndarray[Any, Any],
    rtol: float,
    atol: float,
) -> None:
    """Invariant: f(pad(x, N))[:B] == f(x)."""
    orig_out = _to_numpy(fn(sample_input))
    batch_size = sample_input.shape[0]

    # Pad batch dimension by 2 extra elements
    pad_shape = list(sample_input.shape)
    pad_shape[0] = 2
    padding = np.zeros(pad_shape, dtype=sample_input.dtype)
    padded_input = np.concatenate([sample_input, padding], axis=0)

    padded_out = _to_numpy(fn(padded_input))
    truncated_out = padded_out[:batch_size]

    assert_close(
        truncated_out,
        orig_out,
        rtol=rtol,
        atol=atol,
        err_msg="Metamorphic invariant violation: Batch padding invariance failed.",
    )


def _verify_scale_invariance(
    fn: Callable[..., Any],
    sample_input: np.ndarray[Any, Any],
    rtol: float,
    atol: float,
) -> None:
    """Invariant: f(c * x) == f(x) for c > 0 (e.g. Softmax, LayerNorm)."""
    orig_out = _to_numpy(fn(sample_input))
    scaled_input = sample_input * 2.5
    scaled_out = _to_numpy(fn(scaled_input))

    assert_close(
        scaled_out,
        orig_out,
        rtol=rtol,
        atol=atol,
        err_msg="Metamorphic invariant violation: Scale invariance failed (f(c*x) != f(x)).",
    )


def _verify_scale_equivariance(
    fn: Callable[..., Any],
    sample_input: np.ndarray[Any, Any],
    rtol: float,
    atol: float,
) -> None:
    """Invariant: f(c * x) == c * f(x) (e.g. Linear, Convolution)."""
    c = 2.0
    orig_out = _to_numpy(fn(sample_input))
    expected_out = orig_out * c

    scaled_input = sample_input * c
    actual_out = _to_numpy(fn(scaled_input))

    assert_close(
        actual_out,
        expected_out,
        rtol=rtol,
        atol=atol,
        err_msg="Metamorphic invariant violation: Scale equivariance failed (f(c*x) != c*f(x)).",
    )


def _verify_shift_invariance(
    fn: Callable[..., Any],
    sample_input: np.ndarray[Any, Any],
    rtol: float,
    atol: float,
) -> None:
    """Invariant: f(x + c) == f(x) (e.g. Softmax logits offset)."""
    orig_out = _to_numpy(fn(sample_input))
    shifted_input = sample_input + 10.0
    shifted_out = _to_numpy(fn(shifted_input))

    assert_close(
        shifted_out,
        orig_out,
        rtol=rtol,
        atol=atol,
        err_msg="Metamorphic invariant violation: Shift invariance failed (f(x + c) != f(x)).",
    )


INVARIANT_HANDLERS: dict[
    str, Callable[[Callable[..., Any], np.ndarray[Any, Any], float, float], None]
] = {
    "batch_padding": _verify_batch_padding,
    "scale_invariance": _verify_scale_invariance,
    "scale_equivariance": _verify_scale_equivariance,
    "shift_invariance": _verify_shift_invariance,
}


def test_invariants(
    invariants: Sequence[str] = ("batch_padding",),
    *,
    sample_shape: tuple[int, ...] = (4, 16),
    dtype: str = "float32",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    seed: int = 42,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Test decorator that validates metamorphic invariants on tensor functions or layers.

    Parameters
    ----------
    invariants : Sequence[str]
        List of invariants to verify. Choices:
        - "batch_padding": f(pad(x))[:B] == f(x)
        - "scale_invariance": f(c*x) == f(x) (Softmax, Norms)
        - "scale_equivariance": f(c*x) == c*f(x) (Linear)
        - "shift_invariance": f(x + c) == f(x) (Softmax)
    sample_shape : tuple[int, ...], optional
        Shape of synthetic input tensor to generate if none is passed (default: (4, 16)).
    dtype : str, optional
        Data type of synthetic input (default: "float32").
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    seed : int, optional
        Random seed.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If arguments are passed, use first positional arg as sample input
            if args:
                sample_tensor = _to_numpy(args[0])
            else:
                rng = np.random.default_rng(seed)
                sample_tensor = rng.standard_normal(sample_shape).astype(np.dtype(dtype))

            # Run each specified invariant
            for inv_name in invariants:
                handler = INVARIANT_HANDLERS.get(inv_name)
                if handler is None:
                    supported = list(INVARIANT_HANDLERS.keys())
                    raise ValueError(f"Unknown invariant '{inv_name}'. Supported: {supported}")
                handler(fn, sample_tensor, rtol, atol)

            # Also execute original function if called directly
            if args or kwargs:
                return fn(*args, **kwargs)
            return None

        return wrapper

    return decorator


# Prevent pytest from collecting this helper as a test suite
test_invariants.__test__ = False  # type: ignore[attr-defined]
metamorphic = test_invariants
