"""Automated tolerance calibration for non-flaky CI/CD testing.

Empirically measures the intrinsic stochastic variance of model kernels
(due to multi-threaded reduction nondeterminism, atomic additions, or cuBLAS)
and recommends statistically robust (rtol, atol) boundaries for CI assertions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .assertions import _to_numpy


@dataclass(frozen=True)
class CalibratedTolerances:
    """Calibrated tolerance parameters based on empirical kernel variance."""

    is_deterministic: bool
    recommended_rtol: float
    recommended_atol: float
    observed_max_abs_diff: float
    observed_max_rel_diff: float
    runs_evaluated: int

    def format_summary(self) -> str:
        """Format an informative summary for test authors."""
        det_str = "Deterministic" if self.is_deterministic else "Stochastic (Variance Observed)"
        lines = [
            f"=== XLAxiom Tolerance Calibration: {det_str} ===",
            f"Evaluated runs: {self.runs_evaluated}",
            f"Observed peak absolute variance: {self.observed_max_abs_diff:.4e}",
            f"Observed peak relative variance: {self.observed_max_rel_diff:.4e}",
            "Recommended CI Boundaries (with 2.5x safety headroom):",
            f"  rtol = {self.recommended_rtol:.2e}",
            f"  atol = {self.recommended_atol:.2e}",
            (
                "\nRecommended Assertion:\n"
                f"assert_close(actual, desired, rtol={self.recommended_rtol:.1e}, "
                f"atol={self.recommended_atol:.1e})"
            ),
        ]
        return "\n".join(lines)


def calibrate_tolerances(
    fn: Callable[..., Any],
    *sample_inputs: Any,
    runs: int = 10,
    safety_factor: float = 2.5,
) -> CalibratedTolerances:
    """Empirically measure kernel variance and compute robust test tolerances.

    Parameters
    ----------
    fn : Callable
        Function or model layer to evaluate.
    sample_inputs : Any
        Positional arguments passed to `fn`.
    runs : int, optional
        Number of repeated executions to capture variance (default: 10).
    safety_factor : float, optional
        Multiplier applied to observed peak error to avoid test flakiness (default: 2.5).

    Returns
    -------
    CalibratedTolerances
        Statistical calibration report and recommended (rtol, atol) settings.
    """
    if runs < 2:
        raise ValueError("At least 2 runs are required to calibrate variance.")

    outputs: list[np.ndarray[Any, Any]] = []
    for _ in range(runs):
        res = fn(*sample_inputs)
        val = res[0] if isinstance(res, tuple) else res
        outputs.append(_to_numpy(val).astype(np.float64))

    ref = outputs[0]
    max_abs = 0.0
    max_rel = 0.0

    for out in outputs[1:]:
        abs_diff = np.abs(out - ref)
        mag = np.abs(ref)
        rel_diff = np.zeros_like(abs_diff)
        scalable = mag > 0
        np.divide(abs_diff, mag, out=rel_diff, where=scalable)

        curr_max_abs = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
        curr_max_rel = float(np.max(rel_diff[scalable])) if np.any(scalable) else 0.0

        if curr_max_abs > max_abs:
            max_abs = curr_max_abs
        if curr_max_rel > max_rel:
            max_rel = curr_max_rel

    is_deterministic = max_abs == 0.0

    if is_deterministic:
        # Default tight thresholds for fully deterministic operations
        rec_rtol = 1e-5
        rec_atol = 1e-8
    else:
        # Add safety factor and floor at safe numerical values
        rec_rtol = max(1e-5, max_rel * safety_factor)
        rec_atol = max(1e-8, max_abs * safety_factor)

    return CalibratedTolerances(
        is_deterministic=is_deterministic,
        recommended_rtol=rec_rtol,
        recommended_atol=rec_atol,
        observed_max_abs_diff=max_abs,
        observed_max_rel_diff=max_rel,
        runs_evaluated=runs,
    )
