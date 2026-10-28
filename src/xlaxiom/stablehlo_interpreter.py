"""Pure-Python reference interpreter for StableHLO operations following OpenXLA semantics.

Provides a formal, independent ground-truth oracle directly executing StableHLO
semantics without relying on JAX compiler lowerings or host GPU/TPU device drivers.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .execution import ExecutionResult, ExecutionStatus, compile_execution_plan
from .model import ValidationError, Workload, materialize_inputs


def _stablehlo_gelu(x: NDArray[Any]) -> NDArray[Any]:
    # Canonical tanh approximation for StableHLO chlo.gelu
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi).astype(x.dtype)
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))


def _stablehlo_silu(x: NDArray[Any]) -> NDArray[Any]:
    # StableHLO chlo.silu / swish
    return x / (1.0 + np.exp(-x))


def _stablehlo_relu(x: NDArray[Any]) -> NDArray[Any]:
    zero = np.zeros((), dtype=x.dtype)
    return np.maximum(x, zero)


def _stablehlo_rms_norm(x: NDArray[Any], eps: float = 1e-6) -> NDArray[Any]:
    variance = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps)


def _stablehlo_layer_norm(x: NDArray[Any], eps: float = 1e-6) -> NDArray[Any]:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def _stablehlo_sdpa(q: NDArray[Any], k: NDArray[Any], v: NDArray[Any]) -> NDArray[Any]:
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, v)


class StableHLONamespace:
    """Namespace implementing OpenXLA StableHLO operator semantics."""

    add = staticmethod(np.add)
    subtract = staticmethod(np.subtract)
    multiply = staticmethod(np.multiply)
    divide = staticmethod(np.divide)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    matmul = staticmethod(np.matmul)
    negative = staticmethod(np.negative)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    square = staticmethod(np.square)
    tanh = staticmethod(np.tanh)
    abs = staticmethod(np.abs)
    reshape = staticmethod(np.reshape)
    transpose = staticmethod(np.transpose)
    sum = staticmethod(np.sum)
    mean = staticmethod(np.mean)
    prod = staticmethod(np.prod)
    max = staticmethod(np.max)
    min = staticmethod(np.min)
    gelu = staticmethod(_stablehlo_gelu)
    silu = staticmethod(_stablehlo_silu)
    relu = staticmethod(_stablehlo_relu)
    rms_norm = staticmethod(_stablehlo_rms_norm)
    layer_norm = staticmethod(_stablehlo_layer_norm)
    sdpa = staticmethod(_stablehlo_sdpa)


def interpret_stablehlo(
    workload: Workload,
    inputs: Sequence[NDArray[Any]] | None = None,
) -> ExecutionResult:
    """Execute a workload using the pure-Python StableHLO reference interpreter."""
    started = time.perf_counter()
    try:
        plan = compile_execution_plan(workload)
    except ValidationError as exc:
        return ExecutionResult(
            backend="stablehlo-interpreter",
            mode="reference",
            status=ExecutionStatus.INVALID_WORKLOAD,
            error=str(exc),
        )

    try:
        bound = plan.bind(StableHLONamespace)
    except Exception as exc:
        return ExecutionResult(
            backend="stablehlo-interpreter",
            mode="reference",
            status=ExecutionStatus.UNSUPPORTED_BACKEND,
            error=f"failed to bind StableHLO interpreter: {exc}",
        )

    try:
        if inputs is None:
            raw_inputs = materialize_inputs(workload)
            plan_inputs = tuple(raw_inputs[name] for name in plan.input_names)
        else:
            plan_inputs = tuple(inputs)

        raw_outputs = bound.evaluate(plan_inputs)
        outputs = tuple(np.asarray(val) for val in raw_outputs)
        duration_ms = (time.perf_counter() - started) * 1000.0

        return ExecutionResult(
            backend="stablehlo-interpreter",
            mode="reference",
            status=ExecutionStatus.OK,
            outputs=outputs,
            duration_ms=duration_ms,
            device="interpreter",
            environment={"spec": "OpenXLA StableHLO v1.0", "engine": "pure-python-reference"},
        )
    except Exception:
        return ExecutionResult(
            backend="stablehlo-interpreter",
            mode="reference",
            status=ExecutionStatus.EXECUTION_FAILURE,
            error=traceback.format_exc(),
        )
