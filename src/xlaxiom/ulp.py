"""Units-in-the-last-place (ULP) error measurement.

ULP distance is the number of representable floating-point values between two numbers. Unlike
``rtol``/``atol`` it is meaningful across magnitudes and dtypes: 1 ULP means "adjacent floats"
whether the values are ``1e-30`` or ``1e30``, in float16 or float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .assertions import NumericalAssertionError, _to_numpy

_UINT_BY_SIZE: dict[int, type[np.unsignedinteger[Any]]] = {
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
}


def _is_float_like(arr: np.ndarray[Any, Any]) -> bool:
    return arr.dtype.kind == "f" or arr.dtype.name == "bfloat16"


def _ordered_ints(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Map floats to int64 so that integer order equals float order (and -0 == +0)."""
    bits = np.ascontiguousarray(arr).view(_UINT_BY_SIZE[arr.dtype.itemsize])
    sign_bit = _UINT_BY_SIZE[arr.dtype.itemsize](1 << (arr.dtype.itemsize * 8 - 1))
    magnitude = (bits & ~sign_bit).astype(np.int64)
    negative = (bits & sign_bit) != 0
    result: np.ndarray[Any, Any] = np.where(negative, -magnitude, magnitude)
    return result


def ulp_distance(actual: Any, desired: Any) -> np.ndarray[Any, Any]:
    """Elementwise ULP distance as ``float64`` (``inf`` where exactly one side is NaN).

    Both inputs are compared at the *coarser* of their float precisions (the lower-precision
    value is what determines the achievable accuracy). Non-float inputs are compared in float64.
    Matching NaNs have distance 0. Shapes must match exactly.
    """
    a = _to_numpy(actual)
    b = _to_numpy(desired)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if _is_float_like(a) and _is_float_like(b):
        target = a.dtype if a.dtype.itemsize <= b.dtype.itemsize else b.dtype
        a_f = a.astype(target, copy=False)
        b_f = b.astype(target, copy=False)
    else:
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)

    nan_a = np.isnan(a_f.astype(np.float64))
    nan_b = np.isnan(b_f.astype(np.float64))
    oa = _ordered_ints(a_f)
    ob = _ordered_ints(b_f)
    hi = np.maximum(oa, ob).astype(np.uint64)
    lo = np.minimum(oa, ob).astype(np.uint64)
    with np.errstate(over="ignore"):
        dist = (hi - lo).astype(np.float64)  # modular subtraction is exact for hi >= lo
    dist = np.where(nan_a & nan_b, 0.0, np.where(nan_a | nan_b, np.inf, dist))
    result: np.ndarray[Any, Any] = dist
    return result


def max_ulp_error(actual: Any, desired: Any) -> float:
    """Largest ULP distance between two tensors (0.0 for empty inputs)."""
    dist = ulp_distance(actual, desired)
    return float(np.max(dist)) if dist.size else 0.0


@dataclass(frozen=True)
class UlpStats:
    """Summary of the ULP error distribution between two tensors."""

    max_ulp: float
    mean_ulp: float
    p99_ulp: float
    exact_fraction: float
    total_elements: int

    def format_summary(self) -> str:
        return (
            f"ULP error: max={self.max_ulp:.4g}, mean={self.mean_ulp:.4g}, "
            f"p99={self.p99_ulp:.4g}, bit-exact={self.exact_fraction * 100:.1f}% "
            f"of {self.total_elements} elements"
        )


def ulp_stats(actual: Any, desired: Any) -> UlpStats:
    """Compute :class:`UlpStats` for two tensors."""
    dist = ulp_distance(actual, desired).ravel()
    if dist.size == 0:
        return UlpStats(0.0, 0.0, 0.0, 1.0, 0)
    finite = dist[np.isfinite(dist)]
    mean = float(np.mean(finite)) if finite.size else float("inf")
    return UlpStats(
        max_ulp=float(np.max(dist)),
        mean_ulp=mean if finite.size == dist.size else float("inf"),
        p99_ulp=float(np.percentile(dist, 99)),
        exact_fraction=float(np.mean(dist == 0.0)),
        total_elements=int(dist.size),
    )


def assert_ulp_close(actual: Any, desired: Any, *, max_ulp: float = 4.0) -> None:
    """Assert every element is within ``max_ulp`` ULPs of the reference.

    Raises
    ------
    NumericalAssertionError
        Naming the worst element, its index and both values.
    """
    dist = ulp_distance(actual, desired)
    if dist.size == 0 or float(np.max(dist)) <= max_ulp:
        return
    flat_index = int(np.argmax(dist))
    index = np.unravel_index(flat_index, dist.shape)
    a = _to_numpy(actual)
    b = _to_numpy(desired)
    raise NumericalAssertionError(
        f"ULP assertion failed: {int((dist > max_ulp).sum())}/{dist.size} elements exceed "
        f"{max_ulp:g} ULP. Worst at index {tuple(int(i) for i in index)}: "
        f"actual={a[index]!r}, desired={b[index]!r}, distance={dist[index]:.6g} ULP."
    )

