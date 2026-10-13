"""Small composition layer connecting execution and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .comparison import ComparisonResult, compare
from .execution import ExecutionResult, execute_jax, execute_numpy
from .model import Workload


@dataclass(frozen=True)
class FaultInjection:
    """Explicit test control, never evidence of a JAX/OpenXLA defect."""

    output_offset: float = 0.0
    label: str = "synthetic-control"


@dataclass(frozen=True)
class ExecutionTarget:
    """An explicit execution device request.

    ``platform`` is required, never preferred: if the requested accelerator is absent
    the comparison is reported UNSUPPORTED_BACKEND rather than silently run on CPU.
    """

    platform: str = "cpu"
    device_index: int | None = None


def _kwargs(target: ExecutionTarget | None) -> dict[str, Any]:
    if target is None:
        return {"requested_platform": "cpu"}
    return {"requested_platform": target.platform, "device_index": target.device_index}


def run_comparison(
    workload: Workload,
    mode: str = "jax-jit",
    fault: FaultInjection | None = None,
    target: ExecutionTarget | None = None,
) -> ComparisonResult:
    reference = execute_numpy(workload)
    device = _kwargs(target)
    if mode == "jax-eager":
        candidate = execute_jax(
            workload,
            jit=False,
            output_offset=fault.output_offset if fault else None,
            **device,
        )
    elif mode == "jax-jit":
        candidate = execute_jax(
            workload,
            jit=True,
            output_offset=fault.output_offset if fault else None,
            **device,
        )
    elif mode == "jax-eager-vs-jit":
        reference = execute_jax(workload, jit=False, **device)
        candidate = execute_jax(
            workload,
            jit=True,
            output_offset=fault.output_offset if fault else None,
            **device,
        )
    else:
        raise ValueError(f"unsupported comparison mode {mode}")
    return compare(candidate, reference, workload.policy)


def run_backend(
    workload: Workload, backend: str, target: ExecutionTarget | None = None
) -> ExecutionResult:
    if backend == "numpy":
        return execute_numpy(workload)
    if backend == "jax-eager":
        return execute_jax(workload, **_kwargs(target))
    if backend == "jax-jit":
        return execute_jax(workload, jit=True, **_kwargs(target))
    raise ValueError(f"unsupported backend {backend}")
