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
    DEVICE_MISMATCH = "device_mismatch"


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
    workload: Workload,
    jit: bool = False,
    output_offset: float | None = None,
    requested_platform: str = "cpu",
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
        devices = jax.devices(requested_platform)
        if not devices:
            return ExecutionResult(
                f"jax-{requested_platform}",
                mode,
                ExecutionStatus.UNSUPPORTED_BACKEND,
                error=f"no {requested_platform} device",
            )
        device = devices[0]
        initial = materialize_inputs(workload)
        keys = tuple(spec.name for spec in workload.inputs)

        def program(*args: Any) -> tuple[Any, ...]:
            return _evaluate(workload, dict(zip(keys, args, strict=True)), jnp)

        placed = tuple(jax.device_put(initial[key], device) for key in keys)
        fn = jax.jit(program, device=device) if jit else program
        with jax.default_device(device):
            values = fn(*placed)
        values = tuple(value.block_until_ready() for value in values)
        actual_devices = tuple(str(value.device) for value in values)
        if any(value.device.platform != requested_platform for value in values):
            return ExecutionResult(
                f"jax-{requested_platform}",
                mode,
                ExecutionStatus.DEVICE_MISMATCH,
                error=f"requested {requested_platform}, observed {actual_devices}",
            )
        outputs = tuple(np.asarray(value) for value in values)
        if output_offset is not None:
            # Test-only injection: never inferred from a real compiler execution.
            outputs = (outputs[0] + np.asarray(output_offset, dtype=outputs[0].dtype), *outputs[1:])
        return ExecutionResult(
            f"jax-{requested_platform}",
            mode,
            ExecutionStatus.OK,
            outputs,
            (time.perf_counter() - started) * 1000,
            device=str(device),
            environment={
                "jax": jax.__version__,
                "platform": platform.platform(),
                "requested_platform": requested_platform,
                "actual_devices": ",".join(actual_devices),
            },
        )
    except Exception:
        status = ExecutionStatus.COMPILATION_FAILURE if jit else ExecutionStatus.EXECUTION_FAILURE
        return ExecutionResult(f"jax-{requested_platform}", mode, status, error=traceback.format_exc())


def benchmark_jax(workload: Workload, measurements: int = 10) -> dict[str, float | int | str]:
    """Measure one compilation separately from synchronized steady-state JIT calls."""
    workload.validate()
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("JAX is unavailable") from exc
    device = jax.devices("cpu")[0]
    initial = materialize_inputs(workload)
    keys = tuple(spec.name for spec in workload.inputs)

    def program(*args: Any) -> tuple[Any, ...]:
        return _evaluate(workload, dict(zip(keys, args, strict=True)), jnp)

    placed = tuple(jax.device_put(initial[key], device) for key in keys)
    lowered = jax.jit(program, device=device).lower(*placed)
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_ms = (time.perf_counter() - started) * 1000
    compiled(*placed)[0].block_until_ready()
    samples: list[float] = []
    for _ in range(measurements):
        started = time.perf_counter()
        outputs = compiled(*placed)
        tuple(value.block_until_ready() for value in outputs)
        samples.append((time.perf_counter() - started) * 1000)
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / len(samples)
    return {
        "backend": "jax-cpu-jit",
        "device": str(device),
        "measurements": measurements,
        "compilation_ms": compilation_ms,
        "median_ms": float(np.median(samples)),
        "mean_ms": mean,
        "stdev_ms": variance**0.5,
    }


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
