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
            "environment": self.environment,
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
        axis_value = op.params.get("axis")
        axis = tuple(axis_value) if isinstance(axis_value, list) else axis_value
        return getattr(api, op.op)(args[0], axis=axis, keepdims=op.params.get("keepdims", False))
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


# JAX reports a CUDA device's ``platform`` attribute as ``gpu`` while its name string
# is ``cuda:N``. Availability must match the name prefix; placement must match the
# platform attribute. Mixing the two makes a working GPU look absent or misplaced.
PLATFORM_DEVICE_PREFIX = {"gpu": "cuda", "cpu": "cpu", "tpu": "tpu"}


def _device_prefix(requested_platform: str) -> str:
    """The device-name prefix JAX uses for a requested platform."""
    return PLATFORM_DEVICE_PREFIX.get(requested_platform, requested_platform)


def _platform_present(jax_module: Any, requested_platform: str) -> bool:
    """True when a device of the requested platform is really initialized."""
    prefix = _device_prefix(requested_platform)
    return any(str(device).startswith(prefix) for device in jax_module.devices())


def execute_jax(
    workload: Workload,
    jit: bool = False,
    output_offset: float | None = None,
    requested_platform: str = "cpu",
    device_index: int | None = None,
) -> ExecutionResult:
    """Run JAX eagerly or through ``jax.jit``; JAX must really initialize a device.

    ``requested_platform`` is a hard requirement, not a preference. If the requested
    accelerator platform is absent, this returns UNSUPPORTED_BACKEND and never falls
    back to CPU. ``device_index`` selects an explicit device in that platform's ordinal
    space; placement is verified from the output array's own device afterwards.
    """
    started = time.perf_counter()
    mode = "jit" if jit else "eager"
    try:
        workload.validate()
    except ValidationError as exc:
        return ExecutionResult(
            f"jax-{requested_platform}", mode, ExecutionStatus.INVALID_WORKLOAD, error=str(exc)
        )
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return ExecutionResult(
            f"jax-{requested_platform}", mode, ExecutionStatus.UNSUPPORTED_BACKEND, error=str(exc)
        )
    try:
        available = [str(item) for item in jax.devices()]
    except Exception as exc:
        return ExecutionResult(
            f"jax-{requested_platform}", mode, ExecutionStatus.UNSUPPORTED_BACKEND, error=repr(exc)
        )
    if not _platform_present(jax, requested_platform):
        # Never substitute CPU: report the missing accelerator honestly.
        return ExecutionResult(
            f"jax-{requested_platform}",
            mode,
            ExecutionStatus.UNSUPPORTED_BACKEND,
            error=(
                f"no {requested_platform} device is available; observed {available}. "
                "A CPU fallback was not performed."
            ),
        )
    try:
        devices = jax.devices(requested_platform)
        if not devices:
            return ExecutionResult(
                f"jax-{requested_platform}",
                mode,
                ExecutionStatus.UNSUPPORTED_BACKEND,
                error=f"no {requested_platform} device",
            )
        if device_index is not None:
            if not 0 <= device_index < len(devices):
                return ExecutionResult(
                    f"jax-{requested_platform}",
                    mode,
                    ExecutionStatus.UNSUPPORTED_BACKEND,
                    error=f"device index {device_index} out of range for {len(devices)} devices",
                )
            device = devices[device_index]
        else:
            device = devices[0]
        initial = materialize_inputs(workload)
        keys = tuple(spec.name for spec in workload.inputs)

        def program(*args: Any) -> tuple[Any, ...]:
            return _evaluate(workload, dict(zip(keys, args, strict=True)), jnp)

        placed = tuple(jax.device_put(initial[key], device) for key in keys)
        fn = jax.jit(program) if jit else program
        with jax.default_device(device):
            values = fn(*placed)
        values = tuple(value.block_until_ready() for value in values)
        actual_devices = tuple(str(value.device) for value in values)
        expected_platform = requested_platform
        if any(value.device.platform != expected_platform for value in values):
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
        environment: dict[str, str] = {
            "jax": jax.__version__,
            "platform": platform.platform(),
            "requested_platform": requested_platform,
            "actual_devices": ",".join(actual_devices),
            "device_index": "" if device_index is None else str(device_index),
        }
        return ExecutionResult(
            f"jax-{requested_platform}",
            mode,
            ExecutionStatus.OK,
            outputs,
            (time.perf_counter() - started) * 1000,
            device=str(device),
            environment=environment,
        )
    except Exception:
        status = ExecutionStatus.COMPILATION_FAILURE if jit else ExecutionStatus.EXECUTION_FAILURE
        return ExecutionResult(
            f"jax-{requested_platform}", mode, status, error=traceback.format_exc()
        )


def benchmark_jax(
    workload: Workload,
    measurements: int = 10,
    requested_platform: str = "cpu",
    device_index: int | None = None,
) -> dict[str, float | int | str]:
    """Measure one compilation separately from synchronized steady-state JIT calls.

    ``requested_platform`` is required, not preferred: an unavailable accelerator
    raises rather than silently benchmarking on CPU.
    """
    workload.validate()
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("JAX is unavailable") from exc
    available = [str(item) for item in jax.devices()]
    if not _platform_present(jax, requested_platform):
        raise RuntimeError(
            f"no {requested_platform} device is available; observed {available}. "
            "A CPU fallback was not performed."
        )
    devices = jax.devices(requested_platform)
    if device_index is not None:
        if not 0 <= device_index < len(devices):
            raise RuntimeError(
                f"device index {device_index} out of range for {len(devices)} devices"
            )
        device = devices[device_index]
    else:
        device = devices[0]
    initial = materialize_inputs(workload)
    keys = tuple(spec.name for spec in workload.inputs)

    def program(*args: Any) -> tuple[Any, ...]:
        return _evaluate(workload, dict(zip(keys, args, strict=True)), jnp)

    placed = tuple(jax.device_put(initial[key], device) for key in keys)
    lowered = jax.jit(program).lower(*placed)
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_ms = (time.perf_counter() - started) * 1000
    with jax.default_device(device):
        compiled(*placed)[0].block_until_ready()
    samples: list[float] = []
    for _ in range(measurements):
        started = time.perf_counter()
        with jax.default_device(device):
            outputs = compiled(*placed)
        tuple(value.block_until_ready() for value in outputs)
        samples.append((time.perf_counter() - started) * 1000)
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / len(samples)
    return {
        "backend": f"jax-{requested_platform}-jit",
        "device": str(device),
        "requested_platform": requested_platform,
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
        # Report accelerator capability honestly: a visible but unusable accelerator is
        # recorded as unsupported, never as a CPU result.
        try:
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                raise RuntimeError("no gpu device")
            gpu_probe = jax.device_put(jnp.asarray([1.0], dtype=jnp.float32), gpu_devices[0])
            gpu_probe.block_until_ready()
            info["gpu_devices"] = [str(device) for device in gpu_devices]
            info["gpu_probe"] = "ok"
        except Exception as exc:
            info["gpu_devices"] = []
            info["gpu_probe"] = "unsupported"
            info["gpu_error"] = repr(exc)
    except ImportError as exc:
        info["jax_error"] = str(exc)
    except Exception as exc:
        info["jax_error"] = repr(exc)
    return info
