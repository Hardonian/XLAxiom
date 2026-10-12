"""NumPy and actual JAX execution adapters with typed observations."""

from __future__ import annotations

import platform
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model import Node, ValidationError, Workload, materialize_inputs


class ExecutionStatus(StrEnum):
    OK = "ok"
    INVALID_WORKLOAD = "invalid_workload"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    COMPILATION_FAILURE = "compilation_failure"
    EXECUTION_FAILURE = "execution_failure"


@dataclass(frozen=True)
class ExecutionResult:
    backend: str
    mode: str
    status: ExecutionStatus
    outputs: tuple[NDArray[np.generic], ...] = ()
    duration_ms: float = 0.0
    device: str | None = None
    error: str | None = None
    environment: dict[str, str] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 4),
            "device": self.device,
            "error": self.error,
            "outputs": [
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": int(np.isfinite(value).sum()),
                }
                for value in self.outputs
            ],
        }


def _apply(op: Node, values: dict[str, Any], api: Any) -> Any:
    args = [values[name] for name in op.inputs]
    functions: dict[str, Callable[..., Any]] = {
        "add": api.add,
        "sub": api.subtract,
        "mul": api.multiply,
        "div": api.divide,
        "matmul": api.matmul,
        "neg": api.negative,
        "sin": api.sin,
        "cos": api.cos,
        "exp": api.exp,
        "tanh": api.tanh,
        "abs": api.abs,
    }
    if op.op in functions:
        return functions[op.op](*args)
    if op.op == "reshape":
        return api.reshape(args[0], tuple(op.params["shape"]))
    if op.op == "transpose":
        axes = tuple(op.params.get("axes", tuple(reversed(range(args[0].ndim)))))
        return api.transpose(args[0], axes=axes)
    if op.op in {"sum", "mean"}:
        return getattr(api, op.op)(
            args[0], axis=op.params.get("axis"), keepdims=op.params.get("keepdims", False)
        )
    raise AssertionError(f"registry validation should have rejected {op.op}")


def _evaluate(workload: Workload, initial: dict[str, Any], api: Any) -> tuple[Any, ...]:
    values = dict(initial)
    for node in workload.nodes:
        values[node.output] = _apply(node, values, api)
    return tuple(values[name] for name in workload.outputs)


def execute_numpy(workload: Workload) -> ExecutionResult:
    started = time.perf_counter()
    try:
        workload.validate()
        outputs = tuple(
            np.asarray(value) for value in _evaluate(workload, materialize_inputs(workload), np)
        )
        return ExecutionResult(
            "numpy", "eager", ExecutionStatus.OK, outputs, (time.perf_counter() - started) * 1000
        )
    except ValidationError as exc:
        return ExecutionResult("numpy", "eager", ExecutionStatus.INVALID_WORKLOAD, error=str(exc))
    except Exception:
        return ExecutionResult(
            "numpy", "eager", ExecutionStatus.EXECUTION_FAILURE, error=traceback.format_exc()
        )


def execute_jax(
    workload: Workload, jit: bool = False, output_offset: float | None = None
) -> ExecutionResult:
    """Run JAX eagerly or through ``jax.jit``; JAX must really initialize a device."""
    started = time.perf_counter()
    mode = "jit" if jit else "eager"
    try:
        workload.validate()
    except ValidationError as exc:
        return ExecutionResult("jax-cpu", mode, ExecutionStatus.INVALID_WORKLOAD, error=str(exc))
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return ExecutionResult("jax-cpu", mode, ExecutionStatus.UNSUPPORTED_BACKEND, error=str(exc))
    try:
        devices = jax.devices("cpu")
        if not devices:
            return ExecutionResult(
                "jax-cpu", mode, ExecutionStatus.UNSUPPORTED_BACKEND, error="no CPU device"
            )
        initial = materialize_inputs(workload)
        keys = tuple(spec.name for spec in workload.inputs)

        def program(*args: Any) -> tuple[Any, ...]:
            return _evaluate(workload, dict(zip(keys, args, strict=True)), jnp)

        fn = jax.jit(program) if jit else program
        values = fn(*(jnp.asarray(initial[key]) for key in keys))
        values = tuple(value.block_until_ready() for value in values)
        outputs = tuple(np.asarray(value) for value in values)
        if output_offset is not None:
            # Test-only injection: never inferred from a real compiler execution.
            outputs = (outputs[0] + np.asarray(output_offset, dtype=outputs[0].dtype), *outputs[1:])
        return ExecutionResult(
            "jax-cpu",
            mode,
            ExecutionStatus.OK,
            outputs,
            (time.perf_counter() - started) * 1000,
            device=str(devices[0]),
            environment={"jax": jax.__version__, "platform": platform.platform()},
        )
    except Exception:
        status = ExecutionStatus.COMPILATION_FAILURE if jit else ExecutionStatus.EXECUTION_FAILURE
        return ExecutionResult("jax-cpu", mode, status, error=traceback.format_exc())


def doctor() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": None,
        "devices": [],
    }
    try:
        import jax
        import jax.numpy as jnp
        import jaxlib

        info["jax"] = jax.__version__
        info["jaxlib"] = jaxlib.__version__
        info["devices"] = [str(device) for device in jax.devices()]
        probe = jnp.asarray([1.0], dtype=jnp.float32) + 1.0
        probe.block_until_ready()
        info["cpu_probe"] = "ok"
    except ImportError as exc:
        info["jax_error"] = str(exc)
    except Exception as exc:
        info["jax_error"] = repr(exc)
    return info
