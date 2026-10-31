"""Differential fuzzing and shape-minimization engine for custom GPU/TPU kernels.

Specifically designed for Triton, JAX Pallas, Mojo, CUDA, and custom ops.
Automatically searches for edge-case failures across prime-numbered dimensions,
boundary masking conditions, subnormals, and non-finite values, and shrinks
failing tensors to minimal, copy-pasteable reproduction scripts.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .assertions import NumericalDiff, diff


@dataclass(frozen=True)
class FuzzTrial:
    """Record of a single fuzzer execution trial."""

    trial_index: int
    shape: tuple[int, ...]
    dtype: str
    distribution: str
    passed: bool
    diff: NumericalDiff | None = None
    error: str | None = None


@dataclass(frozen=True)
class KernelFuzzResult:
    """Outcome of differential kernel fuzzing across edge-case shapes and distributions."""

    passed: bool
    trials_run: int
    trials: tuple[FuzzTrial, ...]
    failing_shape: tuple[int, ...] | None = None
    minimal_failing_shape: tuple[int, ...] | None = None
    minimal_input: Any | None = None
    diff: NumericalDiff | None = None
    repro_script: str | None = None

    def format_summary(self) -> str:
        """Format terminal diagnostic report."""
        if self.passed:
            return (
                f"=== Kernel Differential Fuzzing: PASSED ===\n"
                f"Total trials: {self.trials_run} across dynamic shapes & distributions.\n"
                f"Zero discrepancies or boundary crashes observed."
            )
        lines = [
            "=== Kernel Differential Fuzzing: DISCREPANCY DETECTED ===",
            f"Trials completed before failure: {self.trials_run}",
            f"Original failing shape: {self.failing_shape}",
            f"Minimal minimized shape: {self.minimal_failing_shape}",
        ]
        if self.diff:
            lines.append(f"Root-cause diagnosis: {self.diff.diagnosis}")
            lines.append(f"Max absolute error: {self.diff.max_absolute_error:.3e}")
            lines.append(f"Max relative error: {self.diff.max_relative_error:.3e}")
        if self.repro_script:
            lines.append("\n--- Minimal Standalone Reproduction Script ---")
            lines.append(self.repro_script)
        return "\n".join(lines)


def _sample_tensor(
    shape: tuple[int, ...],
    dtype: str,
    distribution: str,
    rng: np.random.Generator,
) -> np.ndarray[Any, Any]:
    """Generate synthetic tensors with specific numerical profiles."""
    np_dtype = np.dtype(dtype)

    if distribution == "normal":
        data = rng.standard_normal(size=shape)
    elif distribution == "uniform":
        data = rng.uniform(-2.0, 2.0, size=shape)
    elif distribution == "extreme":
        # Mix of standard scale, tiny values, large values, and zero
        data = rng.standard_normal(size=shape)
        if data.size > 0:
            flat = data.ravel()
            # Inject extreme finite magnitudes
            n_special = max(1, data.size // 10)
            indices = rng.choice(data.size, size=n_special, replace=False)
            for idx in indices[: n_special // 3]:
                flat[idx] = 1e-15  # near subnormal / tiny
            for idx in indices[n_special // 3 : 2 * (n_special // 3)]:
                flat[idx] = 1e4  # large dynamic range
            for idx in indices[2 * (n_special // 3) :]:
                flat[idx] = 0.0  # exact zero
    elif distribution == "nonfinite":
        data = rng.standard_normal(size=shape)
        if data.size > 0:
            flat = data.ravel()
            flat[0] = np.nan
            if data.size > 1:
                flat[1] = np.inf
    elif distribution == "subnormal":
        data = np.zeros(shape=shape, dtype=np.float32)
        if data.size > 0:
            flat = data.ravel()
            flat[0] = 1e-38  # float32 subnormal range
    else:
        data = rng.standard_normal(size=shape)

    return data.astype(np_dtype)


def _resolve_dimension(spec: int | tuple[int, int] | Sequence[int], rng: random.Random) -> int:
    """Resolve a dimension from a fixed integer, range (min, max), or discrete choice list."""
    if isinstance(spec, int):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2:
        # Range: pick random int with bias towards prime numbers and powers of 2
        low, high = spec
        # Choice between a random prime, power of 2, or uniform
        strategy = rng.choice(["prime", "power2", "uniform"])
        if strategy == "prime":
            primes = [
                p
                for p in [
                    2,
                    3,
                    5,
                    7,
                    11,
                    13,
                    17,
                    19,
                    23,
                    29,
                    31,
                    37,
                    41,
                    43,
                    47,
                    53,
                    59,
                    61,
                    67,
                    71,
                    73,
                    79,
                    83,
                    89,
                    97,
                    127,
                ]
                if low <= p <= high
            ]
            if primes:
                return rng.choice(primes)
        elif strategy == "power2":
            powers = [2**i for i in range(16) if low <= 2**i <= high]
            if powers:
                return rng.choice(powers)
        return rng.randint(low, high)
    if isinstance(spec, Sequence):
        return rng.choice(list(spec))
    return int(spec)


def fuzz_kernel(
    kernel_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    shape_spec: Sequence[int | tuple[int, int] | Sequence[int]],
    *,
    dtypes: Sequence[str] = ("float32",),
    distributions: Sequence[str] = ("normal", "extreme", "uniform"),
    num_trials: int = 25,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    seed: int = 42,
) -> KernelFuzzResult:
    """Differentially fuzz a custom GPU/TPU kernel against a reference implementation.

    Parameters
    ----------
    kernel_fn : Callable
        The custom kernel under test (e.g. Triton, Pallas, or CUDA binding).
    reference_fn : Callable
        The verified reference implementation (e.g. PyTorch eager or JAX native).
    shape_spec : Sequence of dim specifications
        Specification for each dimension. Each element can be:
        - `int`: Fixed dimension size.
        - `(min_val, max_val)`: Dynamic range, sampled with prime and power-of-2 bias.
        - `list[int]`: Discrete choices (e.g. `[32, 64, 128]`).
    dtypes : Sequence[str], optional
        Data types to evaluate (default: ("float32",)).
    distributions : Sequence[str], optional
        Tensor value generation profiles.
    num_trials : int, optional
        Number of fuzzing trials to execute.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    seed : int, optional
        Deterministic random seed.

    Returns
    -------
    KernelFuzzResult
        Comprehensive fuzzing result, including minimal failing shapes and standalone repro.
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    trials: list[FuzzTrial] = []

    for trial_idx in range(num_trials):
        # Sample shape
        shape = tuple(_resolve_dimension(spec, py_rng) for spec in shape_spec)
        dtype = py_rng.choice(list(dtypes))
        dist = py_rng.choice(list(distributions))

        input_tensor = _sample_tensor(shape, dtype, dist, np_rng)

        try:
            kernel_out = kernel_fn(input_tensor)
            ref_out = reference_fn(input_tensor)
            diff_res = diff(kernel_out, ref_out, rtol=rtol, atol=atol)

            if not diff_res.is_close:
                # Discrepancy found!
                trial = FuzzTrial(
                    trial_index=trial_idx,
                    shape=shape,
                    dtype=dtype,
                    distribution=dist,
                    passed=False,
                    diff=diff_res,
                )
                trials.append(trial)

                # Minimize shape to find minimal reproducing failure
                minimal_shape, minimal_tensor, min_diff = _minimize_failing_shape(
                    kernel_fn, reference_fn, shape, dtype, dist, rtol, atol, np_rng
                )

                repro_code = _format_kernel_repro(minimal_shape, dtype, dist, rtol, atol, min_diff)

                return KernelFuzzResult(
                    passed=False,
                    trials_run=trial_idx + 1,
                    trials=tuple(trials),
                    failing_shape=shape,
                    minimal_failing_shape=minimal_shape,
                    minimal_input=minimal_tensor,
                    diff=min_diff,
                    repro_script=repro_code,
                )

            trials.append(
                FuzzTrial(
                    trial_index=trial_idx,
                    shape=shape,
                    dtype=dtype,
                    distribution=dist,
                    passed=True,
                    diff=diff_res,
                )
            )

        except Exception as exc:
            # Exception / Crash during kernel execution
            trial = FuzzTrial(
                trial_index=trial_idx,
                shape=shape,
                dtype=dtype,
                distribution=dist,
                passed=False,
                error=str(exc),
            )
            trials.append(trial)

            repro_code = (
                f"# Kernel execution exception on shape {shape}\n"
                f"# Error: {exc}\n"
                f"import numpy as np\n"
                f"x = np.zeros({shape}, dtype=np.{dtype})\n"
                f"kernel_fn(x)\n"
            )

            return KernelFuzzResult(
                passed=False,
                trials_run=trial_idx + 1,
                trials=tuple(trials),
                failing_shape=shape,
                minimal_failing_shape=shape,
                minimal_input=input_tensor,
                diff=None,
                repro_script=repro_code,
            )

    return KernelFuzzResult(
        passed=True,
        trials_run=num_trials,
        trials=tuple(trials),
    )


def _minimize_failing_shape(
    kernel_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    initial_shape: tuple[int, ...],
    dtype: str,
    distribution: str,
    rtol: float,
    atol: float,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], np.ndarray[Any, Any], NumericalDiff]:
    """Shrink failing tensor shape dimensions to the minimal reproducing configuration."""
    current_shape = list(initial_shape)
    current_tensor = _sample_tensor(initial_shape, dtype, distribution, rng)
    k_out = kernel_fn(current_tensor)
    r_out = reference_fn(current_tensor)
    best_diff = diff(k_out, r_out, rtol=rtol, atol=atol)

    # Greedily reduce each dimension
    for dim_idx in range(len(current_shape)):
        original_dim = current_shape[dim_idx]
        candidates = [1, 2, 3, 4, 8, 16, 32]
        candidates = [c for c in candidates if c < original_dim]

        for cand in candidates:
            test_shape = list(current_shape)
            test_shape[dim_idx] = cand
            cand_shape = tuple(test_shape)
            cand_tensor = _sample_tensor(cand_shape, dtype, distribution, rng)

            try:
                out_k = kernel_fn(cand_tensor)
                out_r = reference_fn(cand_tensor)
                cand_diff = diff(out_k, out_r, rtol=rtol, atol=atol)
                if not cand_diff.is_close:
                    current_shape[dim_idx] = cand
                    current_tensor = cand_tensor
                    best_diff = cand_diff
                    break
            except Exception:
                current_shape[dim_idx] = cand
                current_tensor = cand_tensor
                break

    return tuple(current_shape), current_tensor, best_diff


def _format_kernel_repro(
    shape: tuple[int, ...],
    dtype: str,
    distribution: str,
    rtol: float,
    atol: float,
    diff_res: NumericalDiff,
) -> str:
    """Format a 10-line standalone Python reproduction snippet."""
    return (
        f"import numpy as np\n"
        f"from xlaxiom import assert_close\n\n"
        f"# Discrepancy: {diff_res.diagnosis}\n"
        f"# Minimal reproducing shape: {shape}, dtype: {dtype}\n"
        f"rng = np.random.default_rng(42)\n"
        f"x = rng.standard_normal({shape}).astype(np.{dtype})\n"
        f"out_kernel = kernel_fn(x)\n"
        f"out_ref = reference_fn(x)\n"
        f"assert_close(out_kernel, out_ref, rtol={rtol}, atol={atol})\n"
    )
