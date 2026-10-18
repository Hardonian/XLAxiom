"""Unit tests for arithmetic intensity and roofline efficiency profiler (Priority 65)."""

from xlaxiom.efficiency import KNOWN_HARDWARE, profile_efficiency
from xlaxiom.model import InputSpec, Node, Workload


def test_elementwise_efficiency_is_memory_bound() -> None:
    # 2 inputs of (200, 200) float32 (320 KB total input), 1 output of (200, 200) float32 (160 KB)
    # Total traffic = 480 KB = 480,000 bytes
    # FLOPs = 40,000
    # Intensity = 40,000 / 480,000 = 0.0833 FLOPs/Byte
    workload = Workload(
        identifier="elementwise-profile",
        seed=1,
        inputs=(
            InputSpec("x", shape=(200, 200), dtype="float32"),
            InputSpec("y", shape=(200, 200), dtype="float32"),
        ),
        nodes=(Node(op="add", inputs=("x", "y"), output="z"),),
        outputs=("z",),
    )

    profile = profile_efficiency(workload, hardware_target="nvidia-v100")
    assert profile.regime == "memory_bound"
    assert profile.total_flops == 40_000
    assert profile.arithmetic_intensity < 1.0
    assert profile.attainable_gflops < KNOWN_HARDWARE["nvidia-v100"].peak_gflops


def test_matmul_efficiency_is_compute_bound() -> None:
    # Matrix multiplication has high reuse:
    # A: [200, 200], B: [200, 200] -> C: [200, 200]
    # FLOPs = 2 * 200 * 200 * 200 = 16,000,000
    # Traffic = 320 KB inputs + 160 KB output = 480,000 bytes
    # Intensity = 16,000,000 / 480,000 = 33.33 FLOPs/Byte
    # V100 ridge point is ~17.4 FLOPs/Byte, so this is compute-bound!
    workload = Workload(
        identifier="matmul-profile",
        seed=2,
        inputs=(
            InputSpec("a", shape=(200, 200), dtype="float32"),
            InputSpec("b", shape=(200, 200), dtype="float32"),
        ),
        nodes=(Node(op="matmul", inputs=("a", "b"), output="c"),),
        outputs=("c",),
    )

    profile = profile_efficiency(workload, hardware_target="nvidia-v100")
    assert profile.regime == "compute_bound"
    assert profile.total_flops == 16_000_000
    assert profile.arithmetic_intensity > KNOWN_HARDWARE["nvidia-v100"].ridge_point
    assert profile.attainable_gflops == KNOWN_HARDWARE["nvidia-v100"].peak_gflops
