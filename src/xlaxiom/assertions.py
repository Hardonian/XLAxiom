"""High-level, ergonomic numerical assertions and diagnostic diffing.

Provides drop-in replacements for ``torch.testing.assert_close`` and
``numpy.testing.assert_allclose`` with root-cause condition diagnosis, signal-to-noise ratio,
cosine similarity, catastrophic cancellation detection, and automatic reproduction generation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


class NumericalAssertionError(AssertionError):
    """Raised when numerical values diverge beyond acceptable tolerances."""

    def __init__(self, message: str, diff: NumericalDiff | None = None) -> None:
        super().__init__(message)
        self.diff = diff


def _to_numpy(x: Any) -> np.ndarray[Any, Any]:
    """Convert a tensor, array, list, or scalar to a contiguous NumPy array."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and callable(x.detach):
        # PyTorch Tensor
        return np.asarray(x.detach().cpu().numpy())
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)


@dataclass(frozen=True)
class NumericalDiff:
    """Detailed diagnostic metrics comparing actual and desired numerical tensors."""

    is_close: bool
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    dtype_a: str
    dtype_b: str
    total_elements: int
    mismatched_elements: int
    mismatch_percentage: float
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    rms_error: float
    snr_db: float
    cosine_similarity: float
    catastrophic_cancellations: int
    subnormal_underflows: int
    sign_flips: int
    nan_mismatches: int
    inf_mismatches: int
    p50_absolute_error: float
    p90_absolute_error: float
    p99_absolute_error: float
    diagnosis: str
    repro_snippet: str

    def format_summary(self, title: str = "XLAxiom Numerical Verification") -> str:
        """Format an informative diagnostic report for console and logs."""
        status = "PASSED" if self.is_close else "FAILED"
        status_icon = "[OK]" if self.is_close else "[FAIL]"
        snr_str = f"{self.snr_db:.2f} dB" if not math.isinf(self.snr_db) else "inf dB (exact)"

        lines = [
            f"=== {title}: {status_icon} {status} ===",
            f"Shape: {self.shape_a} vs {self.shape_b} | Dtype: {self.dtype_a} vs {self.dtype_b}",
            (
                f"Mismatched elements: {self.mismatched_elements} / {self.total_elements} "
                f"({self.mismatch_percentage:.2f}%)"
            ),
            (
                f"Max absolute error: {self.max_absolute_error:.6e} | "
                f"Mean abs: {self.mean_absolute_error:.6e}"
            ),
            (
                f"Max relative error: {self.max_relative_error:.6e} | "
                f"RMS error: {self.rms_error:.6e}"
            ),
            (
                f"Signal-to-Noise Ratio (SNR): {snr_str} | "
                f"Cosine similarity: {self.cosine_similarity:.6f}"
            ),
            (
                f"Error percentiles: p50={self.p50_absolute_error:.2e}, "
                f"p90={self.p90_absolute_error:.2e}, p99={self.p99_absolute_error:.2e}"
            ),
            f"Root-Cause Diagnosis: {self.diagnosis}",
        ]
        if not self.is_close:
            lines.append("--- Minimal Reproduction Snippet ---")
            lines.append(self.repro_snippet)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Interactive HTML badge and table for Jupyter Notebooks and Google Colab."""
        badge_color = "#10b981" if self.is_close else "#ef4444"
        verdict = "PASS" if self.is_close else "DISCREPANCY"
        snr_display = f"{self.snr_db:.1f} dB" if not math.isinf(self.snr_db) else "∞ (exact)"
        card_style = (
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 10px 0; "
            "background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05);"
        )
        badge_style = (
            f"background: {badge_color}; color: #ffffff; padding: 3px 10px; "
            "border-radius: 9999px; font-weight: 700; font-size: 12px; letter-spacing: 0.5px;"
        )
        table_style = (
            "width: 100%; border-collapse: collapse; font-size: 13px; "
            "margin-top: 10px; text-align: left;"
        )
        th_style = (
            "border-bottom: 2px solid #cbd5e1; padding: 6px 8px; font-weight: 600; color: #475569;"
        )
        td_style = "border-bottom: 1px solid #f1f5f9; padding: 6px 8px; color: #1e293b;"

        html = f"""
        <div style="{card_style}">
          <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-weight: 700; color: #0f172a; font-size: 15px;">
              XLAxiom Numerical Verification
            </div>
            <span style="{badge_style}">{verdict}</span>
          </div>
          <p style="margin: 6px 0; font-size: 13px; color: #334155;">
            <strong>Diagnosis:</strong> {self.diagnosis}
          </p>
          <table style="{table_style}">
            <thead>
              <tr>
                <th style="{th_style}">Metric</th>
                <th style="{th_style}">Value</th>
                <th style="{th_style}">Metric</th>
                <th style="{th_style}">Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="{td_style}">Max Abs Error</td>
                <td style="{td_style}">{self.max_absolute_error:.3e}</td>
                <td style="{td_style}">Max Rel Error</td>
                <td style="{td_style}">{self.max_relative_error:.3e}</td>
              </tr>
              <tr>
                <td style="{td_style}">SNR</td>
                <td style="{td_style}">{snr_display}</td>
                <td style="{td_style}">Cosine Similarity</td>
                <td style="{td_style}">{self.cosine_similarity:.6f}</td>
              </tr>
              <tr>
                <td style="{td_style}">Mismatches</td>
                <td style="{td_style}">
                  {self.mismatched_elements} / {self.total_elements}
                  ({self.mismatch_percentage:.1f}%)
                </td>
                <td style="{td_style}">RMS Error</td>
                <td style="{td_style}">{self.rms_error:.3e}</td>
              </tr>
            </tbody>
          </table>
        </div>
        """
        return html


def diff(
    actual: Any,
    desired: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = True,
    check_signed_zero: bool = False,
) -> NumericalDiff:
    """Compute rich diagnostic comparison metrics between two arrays or tensors.

    Parameters
    ----------
    actual : array_like or Tensor
        The evaluated result under test.
    desired : array_like or Tensor
        The ground truth or reference baseline.
    rtol : float, optional
        Relative tolerance (default: 1e-5).
    atol : float, optional
        Absolute tolerance (default: 1e-8).
    equal_nan : bool, optional
        Whether NaNs in identical positions compare equal (default: True).
    check_signed_zero : bool, optional
        Whether +0.0 and -0.0 should be distinguished (default: False).

    Returns
    -------
    NumericalDiff
        Structured comparison object containing error metrics, root-cause diagnosis,
        and reproduction code.
    """
    arr_a = _to_numpy(actual)
    arr_b = _to_numpy(desired)

    shape_a = tuple(arr_a.shape)
    shape_b = tuple(arr_b.shape)
    dtype_a = str(arr_a.dtype)
    dtype_b = str(arr_b.dtype)

    # Check shape compatibility
    if shape_a != shape_b:
        diagnosis = f"Shape mismatch: {shape_a} vs {shape_b}"
        snippet = (
            f"# Shape mismatch reproduction\n"
            f"actual = ... # shape {shape_a}\n"
            f"desired = ... # shape {shape_b}\n"
            f"assert actual.shape == desired.shape"
        )
        return NumericalDiff(
            is_close=False,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            total_elements=arr_a.size,
            mismatched_elements=arr_a.size,
            mismatch_percentage=100.0,
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            mean_absolute_error=float("inf"),
            rms_error=float("inf"),
            snr_db=float("-inf"),
            cosine_similarity=0.0,
            catastrophic_cancellations=0,
            subnormal_underflows=0,
            sign_flips=0,
            nan_mismatches=0,
            inf_mismatches=0,
            p50_absolute_error=float("inf"),
            p90_absolute_error=float("inf"),
            p99_absolute_error=float("inf"),
            diagnosis=diagnosis,
            repro_snippet=snippet,
        )

    total_elements = arr_a.size
    if total_elements == 0:
        return NumericalDiff(
            is_close=True,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            total_elements=0,
            mismatched_elements=0,
            mismatch_percentage=0.0,
            max_absolute_error=0.0,
            max_relative_error=0.0,
            mean_absolute_error=0.0,
            rms_error=0.0,
            snr_db=float("inf"),
            cosine_similarity=1.0,
            catastrophic_cancellations=0,
            subnormal_underflows=0,
            sign_flips=0,
            nan_mismatches=0,
            inf_mismatches=0,
            p50_absolute_error=0.0,
            p90_absolute_error=0.0,
            p99_absolute_error=0.0,
            diagnosis="Empty array; trivial match.",
            repro_snippet="",
        )

    # Cast to float64 for high-precision diagnostic analysis
    a_f64 = arr_a.astype(np.float64)
    b_f64 = arr_b.astype(np.float64)

    # Identify finite / nonfinite masks
    finite_a = np.isfinite(a_f64)
    finite_b = np.isfinite(b_f64)
    both_finite = finite_a & finite_b

    # Absolute difference
    abs_diff = np.zeros_like(a_f64)
    np.subtract(a_f64, b_f64, out=abs_diff, where=both_finite)
    np.abs(abs_diff, out=abs_diff)

    # Tolerances
    tol = atol + rtol * np.abs(b_f64)
    finite_match = abs_diff <= tol

    # Special handling: NaN
    nan_a = np.isnan(a_f64)
    nan_b = np.isnan(b_f64)
    nan_match = (nan_a & nan_b) if equal_nan else np.zeros_like(nan_a, dtype=bool)
    nan_mismatches = int(np.sum(nan_a ^ nan_b))

    # Special handling: Inf
    inf_a = np.isinf(a_f64)
    inf_b = np.isinf(b_f64)
    same_inf_sign = np.signbit(a_f64) == np.signbit(b_f64)
    inf_match = inf_a & inf_b & same_inf_sign
    inf_mismatches = int(np.sum((inf_a | inf_b) & ~inf_match))

    # Signed zero check
    sign_flips = 0
    if check_signed_zero:
        both_zero = (a_f64 == 0.0) & (b_f64 == 0.0)
        sign_flips = int(np.sum(both_zero & (np.signbit(a_f64) != np.signbit(b_f64))))
        signed_zero_match = ~both_zero | (np.signbit(a_f64) == np.signbit(b_f64))
    else:
        signed_zero_match = np.ones_like(both_finite, dtype=bool)

    # Total element agreement
    element_agrees = (both_finite & finite_match & signed_zero_match) | nan_match | inf_match
    mismatched_elements = int(np.sum(~element_agrees))
    mismatch_percentage = (mismatched_elements / total_elements) * 100.0
    is_close = mismatched_elements == 0

    # Relative error
    mag_b = np.abs(b_f64)
    rel_diff = np.zeros_like(abs_diff)
    scalable = both_finite & (mag_b > 0)
    np.divide(abs_diff, mag_b, out=rel_diff, where=scalable)

    # Metrics on finite elements
    finite_abs = abs_diff[both_finite]
    if finite_abs.size > 0:
        max_abs = float(np.max(finite_abs))
        mean_abs = float(np.mean(finite_abs))
        rms_err = float(np.sqrt(np.mean(finite_abs**2)))
        p50 = float(np.percentile(finite_abs, 50))
        p90 = float(np.percentile(finite_abs, 90))
        p99 = float(np.percentile(finite_abs, 99))
    else:
        max_abs = 0.0
        mean_abs = 0.0
        rms_err = 0.0
        p50 = 0.0
        p90 = 0.0
        p99 = 0.0

    finite_rel = rel_diff[scalable]
    max_rel = float(np.max(finite_rel)) if finite_rel.size > 0 else 0.0

    # Signal-to-Noise Ratio (SNR) in dB
    noise_power = float(np.sum(abs_diff[both_finite] ** 2))
    signal_power = float(np.sum(b_f64[both_finite] ** 2))
    if noise_power == 0.0:
        snr_db = float("inf")
    elif signal_power == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = 10.0 * math.log10(signal_power / noise_power)

    # Cosine similarity
    vec_a = a_f64[both_finite].ravel()
    vec_b = b_f64[both_finite].ravel()
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 and norm_b == 0.0:
        cos_sim = 1.0
    elif norm_a == 0.0 or norm_b == 0.0:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cos_sim = max(-1.0, min(1.0, cos_sim))

    # Diagnostic condition analysis
    # 1. Catastrophic cancellation: high relative error with moderate absolute error
    catastrophic = both_finite & (rel_diff > 1e-1) & (abs_diff < 1.0) & (abs_diff > atol)
    catastrophic_count = int(np.sum(catastrophic))

    # 2. Subnormal underflow: non-zero in reference, flushed to 0 in actual
    subnormal = both_finite & (a_f64 == 0.0) & (b_f64 != 0.0) & (np.abs(b_f64) < 1e-30)
    subnormal_count = int(np.sum(subnormal))

    # Root-cause diagnostic sentence
    if is_close:
        if math.isinf(snr_db):
            diagnosis = f"Exact match (SNR: {snr_db:.1f} dB, Cosine Sim: {cos_sim:.6f})"
        else:
            diagnosis = (
                f"Numerically close within tolerances (rtol={rtol}, atol={atol}, "
                f"SNR={snr_db:.1f} dB)"
            )
    elif nan_mismatches > 0:
        diagnosis = f"NaN divergence: {nan_mismatches} elements have mismatched NaN presence"
    elif inf_mismatches > 0:
        diagnosis = (
            f"Infinite divergence: {inf_mismatches} elements have mismatched Infinity values"
        )
    elif sign_flips > 0:
        diagnosis = (
            f"Signed zero discrepancy: {sign_flips} elements differ by sign bit (+0.0 vs -0.0)"
        )
    elif subnormal_count > 0:
        diagnosis = (
            f"Subnormal underflow: {subnormal_count} values flushed to zero in actual tensor"
        )
    elif catastrophic_count > 0:
        diagnosis = (
            f"Catastrophic cancellation: {catastrophic_count} elements experienced precision loss "
            f"(max rel diff: {max_rel:.2e})"
        )
    elif max_rel > 1.0:
        diagnosis = f"Severe numerical drift: relative error exceeded 100% (max rel: {max_rel:.2e})"
    else:
        diagnosis = (
            f"Tolerance exceeded: {mismatched_elements} elements exceed threshold "
            f"(max abs: {max_abs:.2e})"
        )

    # Standalone reproduction script
    snippet = (
        "import numpy as np\n"
        "from xlaxiom import assert_close\n\n"
        f"# Shape: {shape_a}, Dtype: {dtype_a}\n"
        f"actual = ...  # Replace with evaluated function output\n"
        f"desired = ... # Replace with reference baseline output\n\n"
        f"assert_close(actual, desired, rtol={rtol}, atol={atol})\n"
    )

    return NumericalDiff(
        is_close=is_close,
        shape_a=shape_a,
        shape_b=shape_b,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        total_elements=total_elements,
        mismatched_elements=mismatched_elements,
        mismatch_percentage=mismatch_percentage,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        mean_absolute_error=mean_abs,
        rms_error=rms_err,
        snr_db=snr_db,
        cosine_similarity=cos_sim,
        catastrophic_cancellations=catastrophic_count,
        subnormal_underflows=subnormal_count,
        sign_flips=sign_flips,
        nan_mismatches=nan_mismatches,
        inf_mismatches=inf_mismatches,
        p50_absolute_error=p50,
        p90_absolute_error=p90,
        p99_absolute_error=p99,
        diagnosis=diagnosis,
        repro_snippet=snippet,
    )


def assert_close(
    actual: Any,
    desired: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = True,
    check_signed_zero: bool = False,
    err_msg: str | None = None,
    verbose: bool = True,
) -> None:
    """Assert that two tensors/arrays are numerically close, with diagnostic failure reporting.

    Parameters
    ----------
    actual : array_like or Tensor
        The evaluated result under test.
    desired : array_like or Tensor
        The reference baseline.
    rtol : float, optional
        Relative tolerance (default: 1e-5).
    atol : float, optional
        Absolute tolerance (default: 1e-8).
    equal_nan : bool, optional
        Whether NaNs compare equal (default: True).
    check_signed_zero : bool, optional
        Whether to enforce sign bit equality on zeroes (default: False).
    err_msg : str, optional
        Custom error message prefix.
    verbose : bool, optional
        Whether to append the complete diagnostic report to the exception.

    Raises
    ------
    NumericalAssertionError
        If the elements diverge beyond the specified tolerance.
    """
    diff_result = diff(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_signed_zero=check_signed_zero,
    )
    if not diff_result.is_close:
        prefix = f"{err_msg}\n" if err_msg else ""
        detail = diff_result.format_summary() if verbose else diff_result.diagnosis
        raise NumericalAssertionError(f"{prefix}{detail}", diff=diff_result)


@dataclass(frozen=True)
class ParityResult:
    """Result of checking parity between two functions or models on sample inputs."""

    is_close: bool
    outputs_a: tuple[Any, ...]
    outputs_b: tuple[Any, ...]
    diffs: tuple[NumericalDiff, ...]

    def summary(self) -> str:
        status = "PASSED" if self.is_close else "FAILED"
        lines = [f"Function Parity Verification: {status}"]
        for idx, d in enumerate(self.diffs):
            lines.append(f"\n--- Output #{idx} ---")
            lines.append(d.format_summary())
        return "\n".join(lines)


def check_parity(
    fn_a: Callable[..., Any],
    fn_b: Callable[..., Any],
    *sample_inputs: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = True,
) -> ParityResult:
    """Execute two callables with the same inputs and verify output parity.

    Parameters
    ----------
    fn_a : Callable
        First callable (e.g. PyTorch eager function, JAX model, or reference implementation).
    fn_b : Callable
        Second callable (e.g. TorchInductor compiled model, Pallas kernel, or quantized model).
    sample_inputs : Any
        Positional arguments passed to both callables.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional
        Whether NaNs compare equal.

    Returns
    -------
    ParityResult
        Result containing outputs and diffs for each returned tensor.
    """
    out_a = fn_a(*sample_inputs)
    out_b = fn_b(*sample_inputs)

    tuple_a = out_a if isinstance(out_a, tuple) else (out_a,)
    tuple_b = out_b if isinstance(out_b, tuple) else (out_b,)

    if len(tuple_a) != len(tuple_b):
        raise ValueError(
            f"Function arity mismatch: fn_a returned {len(tuple_a)} outputs, "
            f"fn_b returned {len(tuple_b)} outputs"
        )

    diffs = [
        diff(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        for a, b in zip(tuple_a, tuple_b, strict=True)
    ]
    all_close = all(d.is_close for d in diffs)

    return ParityResult(
        is_close=all_close,
        outputs_a=tuple_a,
        outputs_b=tuple_b,
        diffs=tuple(diffs),
    )
