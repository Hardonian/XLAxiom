"""NumPy and actual JAX execution adapters with typed observations."""

from __future__ import annotations

import platform
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model import ValidationError, Workload, materialize_inputs
from .operations import OPERATION_SPECS, OperationSpec


class ExecutionStatus(StrEnum):
    OK = "ok"
    INVALID_WORKLOAD = "invalid_workload"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    COMPILATION_FAILURE = "compilation_failure"
    EXECUTION_FAILURE = "execution_failure"
    DEVICE_MISMATCH = "device_mismatch"


class BackendAdapterError(ValueError):
    """A NumPy-style execution namespace cannot run a validated workload."""


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


@dataclass(frozen=True)
class _PlannedNode:
    """A node after names and JSON parameters have been resolved once."""

    name: str
    spec: OperationSpec
    input_slots: tuple[int, ...]
    reshape_shape: tuple[int, ...] | None = None
    axes: tuple[int, ...] | None = None
    axis: int | tuple[int, ...] | None = None
    keepdims: bool = False


@dataclass(frozen=True)
class _BoundNode:
    apply: Callable[[list[Any]], Any]


@dataclass(frozen=True)
class BoundExecutionPlan:
    """A plan bound to one NumPy-compatible operation namespace."""

    input_names: tuple[str, ...]
    nodes: tuple[_BoundNode, ...]
    output_slots: tuple[int, ...]

    def evaluate(self, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if len(inputs) != len(self.input_names):
            raise BackendAdapterError(
                f"execution plan expects {len(self.input_names)} inputs, received {len(inputs)}"
            )
        values = list(inputs)
        for node in self.nodes:
            values.append(node.apply(values))
        return tuple(values[slot] for slot in self.output_slots)


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated, backend-neutral graph execution plan.

    A plan resolves graph names, parameter containers, and operation metadata once.
    Binding it to an API namespace then resolves each operation once as well, leaving
    the per-node hot path as positional list access and one direct call.
    """

    input_names: tuple[str, ...]
    nodes: tuple[_PlannedNode, ...]
    output_slots: tuple[int, ...]

    def bind(self, api: Any) -> BoundExecutionPlan:
        """Bind a validated plan to a NumPy-compatible operation namespace."""
        functions: dict[str, Callable[..., Any]] = {}
        for node in self.nodes:
            if node.spec.api_name in functions:
                continue
            try:
                candidate = getattr(api, node.spec.api_name, None)
            except Exception as exc:
                raise BackendAdapterError(
                    f"backend adapter could not resolve {node.spec.api_name!r} "
                    f"for operation {node.name!r}: {exc}"
                ) from exc
            if not callable(candidate):
                raise BackendAdapterError(
                    f"backend adapter does not provide {node.spec.api_name!r} "
                    f"for operation {node.name!r}"
                )
            functions[node.spec.api_name] = candidate
        return BoundExecutionPlan(
            self.input_names,
            tuple(_bind_node(node, functions[node.spec.api_name]) for node in self.nodes),
            self.output_slots,
        )


def _bind_node(node: _PlannedNode, operation: Callable[..., Any]) -> _BoundNode:
    """Create one direct executor after all symbolic data has been resolved."""
    first = node.input_slots[0]
    if node.spec.kind in {"elementwise", "matmul"}:
        if node.spec.arity == 1:

            def apply_unary(values: list[Any]) -> Any:
                return operation(values[first])

            return _BoundNode(apply_unary)
        second = node.input_slots[1]

        def apply_binary(values: list[Any]) -> Any:
            return operation(values[first], values[second])

        return _BoundNode(apply_binary)
    if node.spec.kind == "reshape":
        assert node.reshape_shape is not None
        shape = node.reshape_shape

        def apply_reshape(values: list[Any]) -> Any:
            return operation(values[first], shape)

        return _BoundNode(apply_reshape)
    if node.spec.kind == "transpose":
        axes = node.axes

        def apply_transpose(values: list[Any]) -> Any:
            value = values[first]
            selected_axes = axes if axes is not None else tuple(reversed(range(value.ndim)))
            return operation(value, axes=selected_axes)

        return _BoundNode(apply_transpose)
    if node.spec.kind == "reduction":
        axis = node.axis
        keepdims = node.keepdims

        def apply_reduction(values: list[Any]) -> Any:
            return operation(values[first], axis=axis, keepdims=keepdims)

        return _BoundNode(apply_reduction)
    raise AssertionError(f"registry validation should have rejected {node.spec!r}")


def compile_execution_plan(workload: Workload) -> ExecutionPlan:
    """Validate a workload once and turn it into a reusable positional plan."""
    workload.validate()
    slots = {spec.name: index for index, spec in enumerate(workload.inputs)}
    nodes: list[_PlannedNode] = []
    for node in workload.nodes:
        spec = OPERATION_SPECS[node.op]
        input_slots = tuple(slots[name] for name in node.inputs)
        axes: tuple[int, ...] | None = None
        axis: int | tuple[int, ...] | None = None
        reshape_shape: tuple[int, ...] | None = None
        keepdims = False
        if spec.kind == "reshape":
            reshape_shape = tuple(node.params["shape"])
        elif spec.kind == "transpose":
            axes_value = node.params.get("axes")
            axes = tuple(axes_value) if axes_value is not None else None
        elif spec.kind == "reduction":
            axis_value = node.params.get("axis")
            axis = tuple(axis_value) if isinstance(axis_value, (list, tuple)) else axis_value
            keepdims = bool(node.params.get("keepdims", False))
        nodes.append(_PlannedNode(node.op, spec, input_slots, reshape_shape, axes, axis, keepdims))
        slots[node.output] = len(slots)
    return ExecutionPlan(
        tuple(spec.name for spec in workload.inputs),
        tuple(nodes),
        tuple(slots[name] for name in workload.outputs),
    )


def _materialized_plan_inputs(plan: ExecutionPlan, workload: Workload) -> tuple[Any, ...]:
    materialized = materialize_inputs(workload)
    return tuple(materialized[name] for name in plan.input_names)


def execute_array_api(
    workload: Workload,
    api: Any,
    *,
    backend: str = "array-api",
    to_backend: Callable[[NDArray[np.generic]], Any] | None = None,
    to_numpy: Callable[[Any], NDArray[np.generic]] = np.asarray,
) -> ExecutionResult:
    """Execute on a validated NumPy-compatible namespace without framework coupling.

    ``to_backend`` and ``to_numpy`` let callers bridge namespaces that do not accept
    NumPy inputs or cannot be converted through ``numpy.asarray``.  JAX callers should
    use :func:`execute_jax` instead, because it additionally proves requested-device
    placement and preserves the package's fail-closed accelerator contract.
    """
    started = time.perf_counter()
    try:
        plan = compile_execution_plan(workload)
    except ValidationError as exc:
        return ExecutionResult(backend, "eager", ExecutionStatus.INVALID_WORKLOAD, error=str(exc))
    try:
        bound = plan.bind(api)
    except BackendAdapterError as exc:
        return ExecutionResult(
            backend, "eager", ExecutionStatus.UNSUPPORTED_BACKEND, error=str(exc)
        )
    try:
        inputs = _materialized_plan_inputs(plan, workload)
        if to_backend is not None:
            inputs = tuple(to_backend(value) for value in inputs)
        outputs = tuple(np.asarray(to_numpy(value)) for value in bound.evaluate(inputs))
        return ExecutionResult(
            backend,
            "eager",
            ExecutionStatus.OK,
            outputs,
            (time.perf_counter() - started) * 1000,
        )
    except Exception:
        return ExecutionResult(
            backend, "eager", ExecutionStatus.EXECUTION_FAILURE, error=traceback.format_exc()
        )


def execute_numpy(workload: Workload) -> ExecutionResult:
    """Execute through the built-in NumPy adapter."""
    return execute_array_api(workload, np, backend="numpy")


def execute_torch(workload: Workload, device: str = "cpu") -> ExecutionResult:
    """Execute through the optional PyTorch adapter for cross-framework differential testing.

    Fail-closed: if PyTorch or the requested device is unavailable, returns UNSUPPORTED_BACKEND.
    """
    started = time.perf_counter()
    try:
        plan = compile_execution_plan(workload)
    except ValidationError as exc:
        return ExecutionResult("torch", "eager", ExecutionStatus.INVALID_WORKLOAD, error=str(exc))

    try:
        import torch
    except ImportError as exc:
        return ExecutionResult(
            "torch",
            "eager",
            ExecutionStatus.UNSUPPORTED_BACKEND,
            error=f"torch is not installed: {exc}",
        )

    if device.startswith("cuda") and not torch.cuda.is_available():
        return ExecutionResult(
            "torch",
            "eager",
            ExecutionStatus.UNSUPPORTED_BACKEND,
            error=f"torch CUDA device {device!r} is unavailable",
        )

    class _TorchNamespace:
        add = staticmethod(torch.add)
        subtract = staticmethod(torch.subtract)
        multiply = staticmethod(torch.multiply)
        divide = staticmethod(torch.divide)
        maximum = staticmethod(torch.maximum)
        minimum = staticmethod(torch.minimum)
        matmul = staticmethod(torch.matmul)
        negative = staticmethod(torch.negative)
        sin = staticmethod(torch.sin)
        cos = staticmethod(torch.cos)
        exp = staticmethod(torch.exp)
        log = staticmethod(torch.log)
        sqrt = staticmethod(torch.sqrt)
        square = staticmethod(torch.square)
        tanh = staticmethod(torch.tanh)
        abs = staticmethod(torch.abs)
        reshape = staticmethod(torch.reshape)

        @staticmethod
        def transpose(tensor: Any, axes: tuple[int, ...] | None = None) -> Any:
            if axes is None:
                return (
                    torch.transpose(tensor, 0, 1)
                    if tensor.ndim == 2
                    else torch.permute(tensor, tuple(reversed(range(tensor.ndim))))
                )
            return torch.permute(tensor, axes)

        @staticmethod
        def sum(
            tensor: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Any:
            if axis is None:
                return torch.sum(tensor)
            return torch.sum(tensor, dim=axis, keepdim=keepdims)

        @staticmethod
        def mean(
            tensor: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Any:
            if axis is None:
                return torch.mean(tensor)
            return torch.mean(tensor, dim=axis, keepdim=keepdims)

        @staticmethod
        def prod(
            tensor: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Any:
            if axis is None:
                return torch.prod(tensor)
            if isinstance(axis, int):
                return torch.prod(tensor, dim=axis, keepdim=keepdims)
            res = tensor
            for d in sorted(axis, reverse=True):
                res = torch.prod(res, dim=d, keepdim=keepdims)
            return res

        @staticmethod
        def max(
            tensor: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Any:
            if axis is None:
                return torch.amax(tensor)
            return torch.amax(tensor, dim=axis, keepdim=keepdims)

        @staticmethod
        def min(
            tensor: Any, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
        ) -> Any:
            if axis is None:
                return torch.amin(tensor)
            return torch.amin(tensor, dim=axis, keepdim=keepdims)

    try:
        bound = plan.bind(_TorchNamespace)
    except BackendAdapterError as exc:
        return ExecutionResult(
            "torch", "eager", ExecutionStatus.UNSUPPORTED_BACKEND, error=str(exc)
        )

    try:
        raw_inputs = _materialized_plan_inputs(plan, workload)
        torch_device = torch.device(device)
        inputs = tuple(torch.from_numpy(np.asarray(val)).to(torch_device) for val in raw_inputs)
        raw_outputs = bound.evaluate(inputs)
        outputs = tuple(np.asarray(val.detach().cpu().numpy()) for val in raw_outputs)
        return ExecutionResult(
            "torch",
            "eager",
            ExecutionStatus.OK,
            outputs,
            (time.perf_counter() - started) * 1000,
            device=str(device),
        )
    except Exception:
        return ExecutionResult(
            "torch", "eager", ExecutionStatus.EXECUTION_FAILURE, error=traceback.format_exc()
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
    target = requested_platform.lower()
    prefix = _device_prefix(requested_platform).lower()
    for device in jax_module.devices():
        platform_attr = getattr(device, "platform", None)
        if platform_attr is not None and str(platform_attr).lower() == target:
            return True
        if str(device).lower().startswith(prefix):
            return True
    return False


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
        plan = compile_execution_plan(workload)
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
        bound = plan.bind(jnp)
        initial = _materialized_plan_inputs(plan, workload)

        def program(*args: Any) -> tuple[Any, ...]:
            return bound.evaluate(args)

        placed = tuple(jax.device_put(value, device) for value in initial)
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
            "jax_enable_x64": str(bool(getattr(jax.config, "jax_enable_x64", False))).lower(),
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
    if measurements < 1:
        raise ValueError("measurements must be at least 1")
    plan = compile_execution_plan(workload)
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
    bound = plan.bind(jnp)
    initial = _materialized_plan_inputs(plan, workload)

    def program(*args: Any) -> tuple[Any, ...]:
        return bound.evaluate(args)

    placed = tuple(jax.device_put(value, device) for value in initial)
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
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "p90_ms": float(np.percentile(samples, 90)),
        "p95_ms": float(np.percentile(samples, 95)),
        "workload_sha256": workload.digest(),
    }


def doctor() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
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
