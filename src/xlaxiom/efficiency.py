"""Arithmetic intensity and roofline efficiency profiler.

Analyzes tensor workload graphs to quantify FLOP counts, memory traffic footprints,
and arithmetic operational intensity (FLOPs/Byte), categorizing execution regimes
against standard accelerator roofline models (NVIDIA V100/A100/H100, Google TPU v5e, CPU).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .model import (
    DEFAULT_LIMITS,
    OPERATION_SPECS,
    ResourceLimits,
    Workload,
    infer_dtype,
    infer_shape,
)


@dataclass(frozen=True)
class HardwareRoofline:
    """Hardware architecture performance limits."""

    name: str
    peak_gflops: float  # GFLOPs/sec (FP32)
    memory_bandwidth_gbs: float  # GB/sec

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity ridge point (FLOPs/Byte) where compute & memory balance."""
        return self.peak_gflops / self.memory_bandwidth_gbs


KNOWN_HARDWARE: dict[str, HardwareRoofline] = {
    "nvidia-v100": HardwareRoofline(
        name="NVIDIA V100 PCIe (FP32)", peak_gflops=15700.0, memory_bandwidth_gbs=900.0
    ),
    "nvidia-a100": HardwareRoofline(
        name="NVIDIA A100 SXM4 (FP32)", peak_gflops=19500.0, memory_bandwidth_gbs=1555.0
    ),
    "nvidia-h100": HardwareRoofline(
        name="NVIDIA H100 SXM5 (FP32)", peak_gflops=60000.0, memory_bandwidth_gbs=3350.0
    ),
    "google-tpu-v5e": HardwareRoofline(
        name="Google TPU v5e (FP32/Matrix)", peak_gflops=24000.0, memory_bandwidth_gbs=819.0
    ),
    "cpu-avx512": HardwareRoofline(
        name="Generic Modern x86-64 AVX-512 Server", peak_gflops=1500.0, memory_bandwidth_gbs=120.0
    ),
}


@dataclass(frozen=True)
class EfficiencyProfile:
    """Workload operational metrics and roofline classification."""

    total_flops: int
    input_bytes: int
    output_bytes: int
    intermediate_bytes: int
    minimal_memory_traffic_bytes: int
    arithmetic_intensity: float
    hardware_target: str
    hardware_ridge_point: float
    attainable_gflops: float
    regime: str  # "memory_bound" | "compute_bound"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_efficiency(
    workload: Workload,
    hardware_target: str = "nvidia-v100",
    limits: ResourceLimits = DEFAULT_LIMITS,
) -> EfficiencyProfile:
    """Calculate operational FLOPs, memory transfer, and roofline bounds for a workload."""
    workload.validate(limits=limits)
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    np_dtypes: dict[str, np.dtype[Any]] = {}

    input_bytes = 0
    for inp in workload.inputs:
        shapes[inp.name] = inp.shape
        dtypes[inp.name] = inp.dtype
        dt = np.dtype(inp.dtype)
        np_dtypes[inp.name] = dt
        input_bytes += math.prod(inp.shape) * dt.itemsize

    total_flops = 0
    intermediate_bytes = 0

    for node in workload.nodes:
        sh = infer_shape(node, shapes)
        dt = infer_dtype(node, np_dtypes)
        shapes[node.output] = sh
        dtypes[node.output] = dt.name
        np_dtypes[node.output] = dt

        num_elements = math.prod(sh)
        node_bytes = num_elements * dt.itemsize
        intermediate_bytes += node_bytes

        spec = OPERATION_SPECS.get(node.op)
        if spec is None:
            continue

        if spec.kind == "elementwise":
            # Transcendental functions cost more than simple arithmetic
            flop_mult = 2 if node.op in ("exp", "log", "sin", "cos", "tanh", "sqrt") else 1
            total_flops += num_elements * flop_mult
        elif spec.kind == "matmul":
            # [M, K] x [K, N] -> 2 * M * N * K
            in0_sh = shapes[node.inputs[0]]
            in1_sh = shapes[node.inputs[1]]
            m = in0_sh[0]
            k = in0_sh[1]
            n = in1_sh[1]
            total_flops += 2 * m * n * k
        elif spec.kind == "reduction":
            in0_sh = shapes[node.inputs[0]]
            total_flops += math.prod(in0_sh)
        elif spec.kind in ("reshape", "transpose"):
            total_flops += 0

    output_bytes = 0
    for out_name in workload.outputs:
        output_bytes += math.prod(shapes[out_name]) * np_dtypes[out_name].itemsize

    # Minimal theoretical memory traffic: reading inputs + writing final outputs
    minimal_traffic = input_bytes + output_bytes
    if minimal_traffic > 0:
        arithmetic_intensity = total_flops / minimal_traffic
    else:
        arithmetic_intensity = 0.0

    hw = KNOWN_HARDWARE.get(hardware_target, KNOWN_HARDWARE["nvidia-v100"])
    ridge = hw.ridge_point

    if arithmetic_intensity < ridge:
        regime = "memory_bound"
        attainable = arithmetic_intensity * hw.memory_bandwidth_gbs
    else:
        regime = "compute_bound"
        attainable = hw.peak_gflops

    return EfficiencyProfile(
        total_flops=total_flops,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        intermediate_bytes=intermediate_bytes,
        minimal_memory_traffic_bytes=minimal_traffic,
        arithmetic_intensity=arithmetic_intensity,
        hardware_target=hw.name,
        hardware_ridge_point=ridge,
        attainable_gflops=min(attainable, hw.peak_gflops),
        regime=regime,
    )


def profile_efficiency_empirical(
    workload: Workload,
    hardware_target: str = "nvidia-v100",
    iterations: int = 10,
    backend: str = "jax-jit",
    limits: ResourceLimits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Empirically measure runtime and compute achieved FLOPs and bandwidth
    against roofline model.
    """
    profile = profile_efficiency(workload, hardware_target=hardware_target, limits=limits)
    from .execution import execute_jax, execute_numpy

    latencies_ms: list[float] = []
    # Warmup
    if backend == "numpy":
        execute_numpy(workload)
        for _ in range(iterations):
            res = execute_numpy(workload)
            latencies_ms.append(res.duration_ms)
    else:
        execute_jax(workload, jit=True, requested_platform="cpu")
        for _ in range(iterations):
            res = execute_jax(workload, jit=True, requested_platform="cpu")
            latencies_ms.append(res.duration_ms)

    median_ms = float(np.median(latencies_ms))
    duration_s = max(median_ms * 1e-3, 1e-9)

    achieved_gflops = (profile.total_flops / duration_s) / 1e9
    achieved_bandwidth_gbs = (profile.minimal_memory_traffic_bytes / duration_s) / 1e9
    percent_of_peak = (
        (achieved_gflops / profile.attainable_gflops * 100.0)
        if profile.attainable_gflops > 0
        else 0.0
    )


    data = profile.to_dict()
    data.update(
        {
            "empirical": True,
            "measured_iterations": iterations,
            "median_latency_ms": round(median_ms, 4),
            "achieved_gflops": round(achieved_gflops, 4),
            "achieved_bandwidth_gbs": round(achieved_bandwidth_gbs, 4),
            "percent_of_roofline_peak": round(percent_of_peak, 4),
        }
    )
    return data

