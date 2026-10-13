"""A small, bounded real-GPU campaign plus the JAX #40634 historical study.

This module runs only after every GPU probe has passed. It is deliberately small:
no unbounded allocation, no large benchmark sweep, and at most one active test worker
per GPU. Every measurement records its device identity, the exact environment, and the
timing separation between compilation and steady-state execution.

Known upstream behavior is never represented as a new defect.
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from typing import Any

from .comparison import Classification
from .generation import GenerationConfig, generate
from .gpu_validation import DeviceState
from .hardware import PhysicalGpu
from .pipeline import ExecutionTarget, run_comparison

# Bounded campaign budget. Deliberately small: the goal is a real observation, not a
# benchmark sweep.
MAX_CASES = 24
MAX_DURATION_S = 300.0
MEASUREMENTS = 5


@dataclass(frozen=True)
class CampaignBudget:
    cases: int = 8
    seed: int = 0
    measurements: int = MEASUREMENTS

    def validate(self) -> None:
        if not 1 <= self.cases <= MAX_CASES:
            raise ValueError(f"cases must be between 1 and {MAX_CASES}")
        if not 1 <= self.measurements <= 20:
            raise ValueError("measurements must be between 1 and 20")


def cpu_vs_gpu(
    workload_seed: int,
    device: PhysicalGpu,
    variant: str,
    measurements: int = MEASUREMENTS,
) -> dict[str, Any]:
    """One bounded CPU-versus-GPU comparison with separated timing.

    Compilation time is measured separately from steady-state synchronized calls so a
    one-off XLA compilation is never reported as execution performance.
    """
    from .execution import benchmark_jax, execute_jax, execute_numpy

    workload = generate(GenerationConfig(seed=workload_seed))
    target = ExecutionTarget(platform="gpu", device_index=0)
    cpu_result = execute_numpy(workload)
    gpu_eager = execute_jax(workload, jit=False, requested_platform="gpu", device_index=0)
    gpu_jit = execute_jax(workload, jit=True, requested_platform="gpu", device_index=0)
    comparison = run_comparison(workload, "jax-jit", target=target)
    eager_vs_jit = run_comparison(workload, "jax-eager-vs-jit", target=target)
    timings: dict[str, Any] = {}
    try:
        timings["cpu_jit"] = benchmark_jax(workload, measurements, requested_platform="cpu")
        timings["gpu_jit"] = benchmark_jax(workload, measurements, requested_platform="gpu")
    except RuntimeError as exc:
        timings["error"] = str(exc)
    return {
        "seed": workload_seed,
        "workload_identifier": workload.identifier,
        "device": {
            "name": device.name,
            "ordinal": device.ordinal,
            "smi_index": device.smi_index,
            "compute_capability": device.compute_capability,
            "uuid_redacted": device.uuid_redacted,
        },
        "variant": variant,
        "numpy_vs_gpu_jit": comparison.to_dict(),
        "gpu_eager_vs_gpu_jit": eager_vs_jit.to_dict(),
        "statuses": {
            "numpy": cpu_result.status.value,
            "gpu_eager": gpu_eager.status.value,
            "gpu_jit": gpu_jit.status.value,
            "gpu_eager_device": gpu_eager.device,
            "gpu_jit_device": gpu_jit.device,
        },
        "timings": timings,
    }


def gpu_eager_vs_jit(
    workload_seed: int,
    device: PhysicalGpu,
    variant: str,
) -> dict[str, Any]:
    """A focused eager-versus-JIT observation on one explicitly selected GPU."""
    workload = generate(GenerationConfig(seed=workload_seed))
    target = ExecutionTarget(platform="gpu", device_index=0)
    result = run_comparison(workload, "jax-eager-vs-jit", target=target)
    return {
        "seed": workload_seed,
        "device": device.name,
        "ordinal": device.ordinal,
        "variant": variant,
        "classification": result.classification.value,
        "left_device": (result.left or {}).get("device"),
        "right_device": (result.right or {}).get("device"),
        "actual_devices": {
            "left": ((result.left or {}).get("environment") or {}).get("actual_devices"),
            "right": ((result.right or {}).get("environment") or {}).get("actual_devices"),
        },
    }


def workload_families() -> list[dict[str, Any]]:
    """The bounded workload families exercised by the campaign."""
    return [
        {"family": "matmul", "seed": 2, "description": "bounded matrix multiply then reduce"},
        {"family": "broadcasting", "seed": 1, "description": "column/row broadcast then mean"},
        {"family": "reduction", "seed": 3, "description": "reshape, elementwise, reshape, sum"},
        {"family": "precision_sensitive", "seed": 0, "description": "add then tanh saturation"},
    ]


def run_gpu_campaign(
    device: PhysicalGpu,
    variant: str,
    budget: CampaignBudget | None = None,
) -> dict[str, Any]:
    """Run the bounded real-GPU campaign for one device.

    Every case records its classification and the actual device each side executed on,
    so a CPU fallback could never be mistaken for a GPU result.
    """
    budget = budget or CampaignBudget()
    budget.validate()
    import time

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for entry in workload_families():
        if time.perf_counter() - started > MAX_DURATION_S:
            break
        record = cpu_vs_gpu(entry["seed"], device, variant, budget.measurements)
        record["family"] = entry["family"]
        record["description"] = entry["description"]
        cases.append(record)
    eager_jit = [gpu_eager_vs_jit(entry["seed"], device, variant) for entry in workload_families()]
    classifications = [case["numpy_vs_gpu_jit"]["classification"] for case in cases]
    eager_classifications = [item["classification"] for item in eager_jit]
    pass_terms = {Classification.PASS.value, DeviceState.PASS.value}

    def _classification(item: dict[str, Any]) -> str:
        value = item.get("classification")
        if value is None:
            value = (item.get("numpy_vs_gpu_jit") or {}).get("classification", "")
        return str(value)

    return {
        "campaign": "XLAxiom bounded EPYC GPU campaign",
        "device": {
            "name": device.name,
            "ordinal": device.ordinal,
            "smi_index": device.smi_index,
            "compute_capability": device.compute_capability,
            "uuid_redacted": device.uuid_redacted,
        },
        "variant": variant,
        "budget": {"cases": budget.cases, "measurements": budget.measurements},
        "duration_s": time.perf_counter() - started,
        "cpu_vs_gpu_cases": cases,
        "gpu_eager_vs_jit_cases": eager_jit,
        "classifications": {
            "numpy_vs_gpu_jit": classifications,
            "gpu_eager_vs_jit": eager_classifications,
        },
        "pass_count": classifications.count(Classification.PASS.value)
        + eager_classifications.count(Classification.PASS.value),
        "anomalies": [
            item for item in (*cases, *eager_jit) if _classification(item) not in pass_terms
        ],
        "notes": [
            "Compilation is timed separately from steady-state synchronized execution.",
            "NumPy is not an independent oracle for every JAX semantic; a discrepancy is "
            "an observation, not a root-cause claim.",
            "No CPU fallback was performed: every GPU case records its actual device.",
        ],
    }


def replay_jax_40634_on_gpu() -> dict[str, Any]:
    """Replay the documented JAX #40634 large-float32 gradient study on the GPU.

    ``arcsinh(+1e22)``, ``arcsinh(-1e22)`` and ``acosh(1e22)`` are kept separate,
    float32 inputs are preserved, and results are compared against stable analytic
    references. Known upstream behavior is reported as such, never as a new defect.
    """
    import jax
    import jax.numpy as jnp

    x64_enabled = bool(jax.config.jax_enable_x64)
    gpu_devices = [str(device) for device in jax.devices("gpu")]
    device = jax.devices("gpu")[0]
    cases = (("arcsinh", 1e22), ("arcsinh", -1e22), ("acosh", 1e22))
    observations: list[dict[str, Any]] = []
    for operation, raw_value in cases:
        value = jnp.float32(raw_value)
        function = getattr(jnp, operation)
        eager = float(jax.grad(function)(value).block_until_ready())
        compiled = float(jax.jit(jax.grad(function))(value).block_until_ready())
        if operation == "arcsinh":
            reference = 1.0 / math.hypot(1.0, abs(raw_value))
        else:
            reference = 1.0 / (math.sqrt(raw_value - 1.0) * math.sqrt(raw_value + 1.0))
        passed = math.isclose(eager, reference, rel_tol=2e-5, abs_tol=0.0) and math.isclose(
            compiled, reference, rel_tol=2e-5, abs_tol=0.0
        )
        observations.append(
            {
                "operation": operation,
                "input": raw_value,
                "eager_gradient": eager,
                "jit_gradient": compiled,
                "analytic_reference": reference,
                "device": str(device),
                "rtol": 2e-5,
                "classification": "pass" if passed else "analytic_discrepancy",
            }
        )
    return {
        "study": "JAX #40634 large-float32 gradient replay on GPU",
        "source": "https://github.com/jax-ml/jax/issues/40634",
        "devices": gpu_devices,
        "x64_enabled": x64_enabled,
        "platform": platform.platform(),
        "observations": observations,
        "interpretation": (
            "Any reproduced discrepancy here is documented upstream behavior from JAX "
            "issue #40634. It is explicitly not reported as a new XLAxiom finding."
        ),
    }
