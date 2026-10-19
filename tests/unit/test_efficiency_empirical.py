"""Unit tests for empirical roofline efficiency calibration."""

from __future__ import annotations

from xlaxiom.efficiency import profile_efficiency_empirical
from xlaxiom.model import InputSpec, Node, Workload


def _matmul_workload() -> Workload:
    inp1 = InputSpec("x", (32, 32), "float32")
    inp2 = InputSpec("y", (32, 32), "float32")
    node = Node("matmul", ("x", "y"), "out")
    return Workload(
        identifier="efficiency-empirical-workload",
        seed=42,
        inputs=(inp1, inp2),
        nodes=(node,),
        outputs=("out",),
    )


def test_profile_efficiency_empirical() -> None:
    workload = _matmul_workload()
    res = profile_efficiency_empirical(
        workload, hardware_target="google-tpu-v5e", iterations=5, backend="numpy"
    )

    assert res["empirical"] is True
    assert res["measured_iterations"] == 5
    assert res["total_flops"] == 2 * 32 * 32 * 32
    assert "median_latency_ms" in res
    assert "achieved_gflops" in res
    assert "achieved_bandwidth_gbs" in res
    assert "percent_of_roofline_peak" in res
    assert res["hardware_target"] == "Google TPU v5e (FP32/Matrix)"
