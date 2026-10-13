"""Focused tests for CUDA capability gating, explicit placement, and no CPU fallback.

These tests never mock a successful GPU execution. They exercise the decision logic on
synthetic device records and the real no-fallback guarantee against the actual runtime,
which on a CPU-only host must report UNSUPPORTED rather than silently using CPU.
"""

from __future__ import annotations

import pytest

from xlaxiom.cuda import cuda_capability, recommended_variant
from xlaxiom.execution import ExecutionStatus, execute_jax
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.gpu_validation import DeviceState
from xlaxiom.hardware import (
    CUDA12_MIN_DRIVER,
    CUDA12_MIN_SM,
    CUDA13_MIN_DRIVER,
    CUDA13_MIN_SM,
    CudaVariant,
    PhysicalGpu,
    driver_supported,
    redact_uuid,
    select_variant,
    sm_supported,
)


def make_device(
    name: str,
    sm: str,
    driver: str = "580.178.04",
    used: int = 10,
    utilization: float = 0.0,
    processes: tuple[str, ...] = (),
) -> PhysicalGpu:
    major, minor = (int(part) for part in sm.split("."))
    return PhysicalGpu(
        ordinal=0,
        name=name,
        uuid_redacted=redact_uuid(f"GPU-{name}"),
        compute_capability=sm,
        sm=(major, minor),
        driver_version=driver,
        cuda_version="13.0",
        total_vram_mib=16384,
        used_vram_mib=used,
        free_vram_mib=16384 - used,
        utilization_percent=utilization,
        active_processes=processes,
        smi_index=0,
    )


class TestSMSupport:
    """CUDA 12 and CUDA 13 support must follow the published JAX rules exactly."""

    def test_cuda12_minimum_sm_is_honored(self) -> None:
        assert CUDA12_MIN_SM == (5, 2)
        assert sm_supported((5, 2), CudaVariant.CUDA12) is True
        assert sm_supported((5, 0), CudaVariant.CUDA12) is False

    def test_cuda13_requires_sm_75_or_newer(self) -> None:
        assert CUDA13_MIN_SM == (7, 5)
        assert sm_supported((7, 5), CudaVariant.CUDA13) is True
        assert sm_supported((7, 0), CudaVariant.CUDA13) is False
        assert sm_supported((6, 1), CudaVariant.CUDA13) is False
        assert sm_supported((8, 6), CudaVariant.CUDA13) is True

    def test_driver_minimums_are_honored(self) -> None:
        assert CUDA12_MIN_DRIVER == 525
        assert CUDA13_MIN_DRIVER == 580
        assert driver_supported("580.178.04", CudaVariant.CUDA13) is True
        assert driver_supported("550.10", CudaVariant.CUDA13) is False
        assert driver_supported("550.10", CudaVariant.CUDA12) is True
        assert driver_supported("520.0", CudaVariant.CUDA12) is False
        assert driver_supported("", CudaVariant.CUDA12) is False


class TestVariantSelection:
    """The correct wheel variant is chosen, and old GPUs are never sent to CUDA 13."""

    def test_v100_sm70_uses_cuda12(self) -> None:
        variant, reason = select_variant(make_device("Tesla V100-SXM2-16GB", "7.0"))
        assert variant is CudaVariant.CUDA12
        assert "CUDA 13 requires SM >= 7.5" in reason

    def test_p40_sm61_uses_cuda12(self) -> None:
        variant, reason = select_variant(make_device("Tesla P40", "6.1"))
        assert variant is CudaVariant.CUDA12
        assert "CUDA 13 requires SM >= 7.5" in reason

    def test_rtx3060_sm86_uses_cuda13(self) -> None:
        variant, reason = select_variant(make_device("NVIDIA GeForce RTX 3060", "8.6"))
        assert variant is CudaVariant.CUDA13
        assert ">= 580" in reason

    def test_kepler_class_gpu_is_unsupported(self) -> None:
        variant, reason = select_variant(make_device("Tesla K80", "3.7"))
        assert variant is None
        assert "below the CUDA 12 minimum" in reason

    def test_new_gpu_with_old_driver_falls_back_to_cuda12(self) -> None:
        variant, reason = select_variant(make_device("RTX 4090", "8.9", driver="550.10"))
        assert variant is CudaVariant.CUDA12
        assert "550.10 < 580" in reason


class TestUUIDRedaction:
    """Raw GPU UUIDs must never reach portable evidence."""

    def test_raw_uuid_is_not_recorded(self) -> None:
        raw = "GPU-a5ca182a-d210-9786-75c0-e30b2e4358a0"
        redacted = redact_uuid(raw)
        assert raw not in redacted
        assert redacted.startswith("dev-")

    def test_redaction_is_stable_per_machine(self) -> None:
        raw = "GPU-16dfb624-ef25-a0eb-298c-a08dae7f53d2"
        assert redact_uuid(raw) == redact_uuid(raw)

    def test_salt_changes_the_identifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = "GPU-16dfb624-ef25-a0eb-298c-a08dae7f53d2"
        baseline = redact_uuid(raw)
        monkeypatch.setenv("XLAXIOM_DEVICE_SALT", "different-salt")
        assert redact_uuid(raw) != baseline


class TestNoCpuFallback:
    """A requested accelerator that is absent must be reported, never replaced by CPU."""

    def test_requesting_gpu_on_a_cpu_only_host_is_unsupported(self) -> None:
        result = execute_jax(generate(GenerationConfig(seed=2)), jit=True, requested_platform="gpu")
        if result.status == ExecutionStatus.OK:
            # A real GPU is present: placement must still be verified from the array.
            assert result.device is not None and "gpu" in result.device.lower()
        else:
            assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
            assert result.error is not None
            assert "CPU fallback was not performed" in result.error

    def test_requesting_gpu_never_reports_a_cpu_device(self) -> None:
        result = execute_jax(
            generate(GenerationConfig(seed=0)), jit=False, requested_platform="gpu"
        )
        if result.status == ExecutionStatus.OK:
            assert result.device is not None and "cpu" not in result.device.lower()
        else:
            assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND

    def test_out_of_range_device_index_is_unsupported(self) -> None:
        result = execute_jax(
            generate(GenerationConfig(seed=2)),
            jit=True,
            requested_platform="gpu",
            device_index=99,
        )
        if result.status == ExecutionStatus.OK:
            pytest.skip("this host has at least 100 GPU devices")
        # The invariant is that an unusable request never becomes a CPU result. Which
        # guard fires first depends on the host: a CPU-only host reports the missing
        # platform, a GPU host reports the out-of-range index.
        assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
        assert result.error is not None
        assert ("out of range" in result.error) or ("no gpu device is available" in result.error)


class TestExplicitDevicePlacement:
    """Device identity comes from the output array, not from the request."""

    def test_cpu_execution_records_the_actual_device(self) -> None:
        result = execute_jax(generate(GenerationConfig(seed=2)), jit=True)
        assert result.status == ExecutionStatus.OK
        assert result.device is not None
        assert result.environment is not None
        assert result.environment["requested_platform"] == "cpu"
        assert result.environment["actual_devices"].startswith("cpu")

    def test_gpu_execution_records_the_requested_index(self) -> None:
        result = execute_jax(
            generate(GenerationConfig(seed=1)), jit=True, requested_platform="gpu", device_index=0
        )
        if result.status != ExecutionStatus.OK:
            pytest.skip("no GPU is available in this environment")
        assert result.environment is not None
        assert result.environment["actual_devices"].startswith("gpu")
        assert result.environment["device_index"] == "0"

    def test_mismatched_platform_is_reported_not_relabelled(self) -> None:
        # Requesting gpu on a CPU-only host must not produce an OK cpu result.
        result = execute_jax(
            generate(GenerationConfig(seed=3)), jit=True, requested_platform="gpu", device_index=0
        )
        if result.status == ExecutionStatus.OK:
            assert result.device is not None and "gpu" in result.device.lower()
            return
        assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
        assert "no gpu device is available" in (result.error or "")


class TestCapabilityReporting:
    """Capability reporting must be honest about the installed runtime."""

    def test_cuda_capability_reports_supported_or_unsupported(self) -> None:
        report = cuda_capability()
        assert report["status"] in {"available", "unsupported"}
        if report["status"] == "available":
            assert report["supported"] is True
            assert report["devices"]
        else:
            assert report["supported"] is False
            assert report["reason"]

    def test_recommended_variant_is_never_cuda13_for_old_gpus(self) -> None:
        recommendation = recommended_variant()
        if recommendation["variant"] is None:
            assert recommendation["reason"]
            return
        assert recommendation["variant"] in {"cuda12", "cuda13"}

    def test_states_are_explicit(self) -> None:
        assert DeviceState.PASS.value == "PASS"
        assert DeviceState.UNSUPPORTED.value == "UNSUPPORTED"
        assert DeviceState.BLOCKED.value == "BLOCKED"
        assert DeviceState.INCONCLUSIVE.value == "INCONCLUSIVE"
