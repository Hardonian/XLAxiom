import time
import unittest.mock as mock

from xlaxiom.cuda import recommended_variant
from xlaxiom.hardware import PhysicalGpu


def make_device(
    name: str, capability: str, driver: str = "525.60", ordinal: int = 0
) -> PhysicalGpu:
    sm = (int(capability.split(".")[0]), int(capability.split(".")[1]))
    return PhysicalGpu(
        ordinal=ordinal,
        name=name,
        uuid_redacted="dev-test",
        compute_capability=capability,
        sm=sm,
        driver_version=driver,
        cuda_version="12.6",
        total_vram_mib=24576,
        used_vram_mib=100,
        free_vram_mib=24476,
        utilization_percent=0.0,
    )


def run_benchmark() -> None:
    # Scenario 1: CUDA 12 devices
    cuda12_devices = [make_device(f"GPU-{i}", "7.0", driver="525.60", ordinal=i) for i in range(50)]

    # Scenario 2: unsupported devices
    unsupported_devices = [
        make_device(f"GPU-{i}", "3.7", driver="470.00", ordinal=i) for i in range(50)
    ]

    # Scenario 3: Mixed devices (49 CUDA 12 devices, 1 CUDA 13 device at end)
    mixed_devices = [make_device(f"GPU-{i}", "7.0", driver="525.60", ordinal=i) for i in range(49)]
    mixed_devices.append(make_device("GPU-49", "9.0", driver="580.30", ordinal=49))

    scenarios = [
        ("CUDA 12 GPUs (50 devices)", cuda12_devices),
        ("Unsupported GPUs (50 devices)", unsupported_devices),
        ("Mixed GPUs (49 CUDA12 + 1 CUDA13)", mixed_devices),
    ]

    iterations = 20000
    print(f"Running benchmarks ({iterations} iterations per scenario)...")
    for name, devs in scenarios:
        with mock.patch("xlaxiom.cuda.enumerate_physical_gpus", return_value=devs):
            t0 = time.perf_counter()
            for _ in range(iterations):
                recommended_variant()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"  {name}: {elapsed:.4f} seconds ({elapsed / iterations * 1e6:.2f} us/call)")


if __name__ == "__main__":
    run_benchmark()
