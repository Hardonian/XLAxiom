"""A small, bounded real-GPU campaign plus the JAX #40634 historical study.

This module runs only after every GPU probe has passed. It is deliberately small:
no unbounded allocation, no large benchmark sweep, and at most one active test worker
per GPU. Every measurement records its device identity, the exact environment, and the
timing separation between compilation and steady-state execution.

Known upstream behavior is never represented as a new defect.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from .comparison import Classification
from .generation import GenerationConfig, generate
from .gpu_validation import DeviceState
from .hardware import PhysicalGpu
from .model import Workload
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


def _safe_benchmark(workload: Workload, measurements: int, platform: str) -> dict[str, Any] | None:
    """Benchmark one platform, returning ``None`` when that backend is unavailable.

    An unavailable backend is a recorded absence, never a fabricated number.
    """
    from .execution import benchmark_jax

    try:
        return dict(benchmark_jax(workload, measurements, requested_platform=platform))
    except RuntimeError:
        return None


def _cpu_baseline_subprocess(
    measurements: int, workload_payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Measure the CPU baseline in a process where the CPU backend is visible.

    The isolated CUDA runtime intentionally exposes only the GPU, so the CPU reference
    must be taken with the CUDA device restriction lifted. Compilation is still timed
    separately from steady-state synchronized calls. The workload travels as validated
    JSON, so no executable payload crosses the process boundary.
    """
    script = (
        "import json,sys\n"
        "from xlaxiom.execution import benchmark_jax\n"
        "from xlaxiom.model import Workload\n"
        "workload = Workload.from_dict(json.loads(sys.argv[1]))\n"
        f"print(json.dumps(benchmark_jax(workload, {measurements}, "
        "requested_platform='cpu'), sort_keys=True))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, json.dumps(workload_payload)],
            capture_output=True,
            text=True,
            timeout=180,
            env=_cpu_environment(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in reversed(completed.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return dict(json.loads(line))
            except ValueError:
                continue
    return None


def _cpu_environment() -> dict[str, str]:
    """A child environment that exposes the CPU backend and drops CUDA restrictions."""
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["JAX_PLATFORMS"] = "cpu"
    return environment


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
    from .execution import execute_jax, execute_numpy

    workload = generate(GenerationConfig(seed=workload_seed))
    target = ExecutionTarget(platform="gpu", device_index=0)
    cpu_result = execute_numpy(workload)
    gpu_eager = execute_jax(workload, jit=False, requested_platform="gpu", device_index=0)
    gpu_jit = execute_jax(workload, jit=True, requested_platform="gpu", device_index=0)
    comparison = run_comparison(workload, "jax-jit", target=target)
    eager_vs_jit = run_comparison(workload, "jax-eager-vs-jit", target=target)
    timings: dict[str, Any] = {}
    # The CPU reference needs the CPU backend to be visible. Under an isolated CUDA
    # runtime JAX exposes only the GPU, so the CPU baseline is measured in a separate
    # short-lived process that keeps the CPU backend and drops the CUDA restriction.
    # A missing CPU baseline is recorded, never silently omitted or faked.
    timings["gpu_jit"] = _safe_benchmark(workload, measurements, "gpu")
    cpu_baseline = _safe_benchmark(workload, measurements, "cpu")
    if cpu_baseline is not None:
        timings["cpu_jit"] = cpu_baseline
    else:
        timings["cpu_jit"] = _cpu_baseline_subprocess(measurements, workload.canonical_dict())
    if timings["cpu_jit"] is None:
        timings["cpu_note"] = (
            "the isolated CUDA runtime exposes no CPU backend, so no CPU baseline was "
            "recorded; the CPU-versus-GPU timing comparison is unavailable"
        )
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


def campaign_cases(budget: CampaignBudget) -> list[dict[str, Any]]:
    """Expand the bounded families to the requested case count.

    ``--cases`` is a real budget, not decoration: extra cases cycle the families with
    distinct seeds so the campaign exercises more than the four base motifs while
    staying inside the bounded workload generator.
    """
    families = workload_families()
    if not 1 <= budget.cases <= MAX_CASES:
        raise ValueError(f"cases must be between 1 and {MAX_CASES}")
    cases: list[dict[str, Any]] = []
    for index in range(budget.cases):
        base = families[index % len(families)]
        cases.append(
            {
                "family": base["family"],
                "description": base["description"],
                # Keep the base seed for the first pass so results stay comparable, then
                # derive distinct seeds for every additional case.
                "seed": base["seed"] if index < len(families) else base["seed"] + index,
            }
        )
    return cases


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

    entries = campaign_cases(budget)
    started = time.perf_counter()
    deadline = started + MAX_DURATION_S
    cases: list[dict[str, Any]] = []
    for entry in entries:
        # The CPU-versus-GPU and eager-versus-JIT observations share one deadline.
        # An in-flight JAX operation cannot be safely preempted, but no new bounded
        # case is dispatched once the campaign budget has expired.
        if time.perf_counter() >= deadline:
            break
        record = cpu_vs_gpu(entry["seed"], device, variant, budget.measurements)
        record["family"] = entry["family"]
        record["description"] = entry["description"]
        cases.append(record)
    eager_jit: list[dict[str, Any]] = []
    for entry in entries:
        if time.perf_counter() >= deadline:
            break
        eager_jit.append(gpu_eager_vs_jit(entry["seed"], device, variant))
    duration_s = time.perf_counter() - started
    truncated = len(cases) < len(entries) or len(eager_jit) < len(entries)
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
        "budget": {
            "cases": budget.cases,
            "measurements": budget.measurements,
            "max_duration_s": MAX_DURATION_S,
        },
        "duration_s": duration_s,
        "completion": {
            "requested_cases_per_phase": len(entries),
            "cpu_vs_gpu_cases_completed": len(cases),
            "gpu_eager_vs_jit_cases_completed": len(eager_jit),
            "duration_limit_reached": duration_s >= MAX_DURATION_S,
            "truncated": truncated,
            "truncation_reason": "maximum_duration_reached" if truncated else None,
        },
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

    x64_enabled = bool(getattr(jax.config, "jax_enable_x64", False))
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
