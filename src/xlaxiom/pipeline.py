"""Small composition layer connecting execution and comparison."""

from __future__ import annotations

from dataclasses import dataclass

from .comparison import ComparisonResult, compare
from .execution import ExecutionResult, execute_jax, execute_numpy
from .model import Workload


@dataclass(frozen=True)
class FaultInjection:
    """Explicit test control, never evidence of a JAX/OpenXLA defect."""

    output_offset: float = 0.0
    label: str = "synthetic-control"


def run_comparison(
    workload: Workload, mode: str = "jax-jit", fault: FaultInjection | None = None
) -> ComparisonResult:
    reference = execute_numpy(workload)
    if mode == "jax-eager":
        candidate = execute_jax(
            workload, jit=False, output_offset=fault.output_offset if fault else None
        )
    elif mode == "jax-jit":
        candidate = execute_jax(
            workload, jit=True, output_offset=fault.output_offset if fault else None
        )
    elif mode == "jax-eager-vs-jit":
        reference = execute_jax(workload, jit=False)
        candidate = execute_jax(
            workload, jit=True, output_offset=fault.output_offset if fault else None
        )
    else:
        raise ValueError(f"unsupported comparison mode {mode}")
    return compare(candidate, reference, workload.policy)


def run_backend(workload: Workload, backend: str) -> ExecutionResult:
    if backend == "numpy":
        return execute_numpy(workload)
    if backend == "jax-eager":
        return execute_jax(workload)
    if backend == "jax-jit":
        return execute_jax(workload, jit=True)
    raise ValueError(f"unsupported backend {backend}")
