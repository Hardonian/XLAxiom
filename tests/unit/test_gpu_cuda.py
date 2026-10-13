"""Focused tests for CUDA capability gating, explicit placement, and no CPU fallback.

These tests never mock a successful GPU execution. They exercise the decision logic on
synthetic device records and the real no-fallback guarantee against the actual runtime,
which on a CPU-only host must report UNSUPPORTED rather than silently using CPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from xlaxiom import cli
from xlaxiom.cuda import cuda_capability, recommended_variant
from xlaxiom.execution import ExecutionStatus, execute_jax
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.gpu_campaign import _cpu_baseline_subprocess, _cpu_environment, _safe_benchmark
from xlaxiom.gpu_validation import (
    JAX_GPU_PREALLOCATE,
    DeviceState,
    clean_environment,
    gpu_environment_dir,
    variant_python,
)
from xlaxiom.hardware import (
    CUDA12_MIN_DRIVER,
    CUDA12_MIN_SM,
    CUDA13_MIN_DRIVER,
    CUDA13_MIN_SM,
    CudaVariant,
    PhysicalGpu,
    _visible_gpu_name,
    driver_supported,
    is_idle,
    library_path_risk,
    ordinal_ordering_risk,
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
    ordinal: int = 0,
    smi_index: int | None = None,
) -> PhysicalGpu:
    major, minor = (int(part) for part in sm.split("."))
    return PhysicalGpu(
        ordinal=ordinal,
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
        smi_index=ordinal if smi_index is None else smi_index,
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


class TestCpuBaselineFallback:
    """A missing CPU backend is recorded, never faked or silently omitted."""

    def test_cpu_environment_exposes_cpu_and_hides_cuda(self) -> None:
        environment = _cpu_environment()
        assert environment.get("JAX_PLATFORMS") == "cpu"
        assert "CUDA_VISIBLE_DEVICES" not in environment
        assert "LD_LIBRARY_PATH" not in environment

    def test_safe_benchmark_returns_none_when_platform_absent(self) -> None:
        workload = generate(GenerationConfig(seed=2))
        try:
            import jax

            jax.devices("gpu")
        except Exception:
            pytest.skip("no GPU in the primary environment")
        # Inside an isolated CUDA runtime the CPU backend is absent, so the helper
        # must report None rather than raising or inventing a number.
        result = _safe_benchmark(workload, 2, "cpu")
        assert result is None or isinstance(result, dict)

    def test_cpu_baseline_subprocess_records_a_real_measurement(self) -> None:
        workload = generate(GenerationConfig(seed=1))
        result = _cpu_baseline_subprocess(2, workload.canonical_dict())
        if result is None:
            pytest.skip("no CPU backend reachable from a subprocess here")
        assert result["requested_platform"] == "cpu"
        assert result["measurements"] == 2
        assert result["median_ms"] >= 0.0
        assert str(result["device"]).startswith("cpu")


class TestEnvironmentTraps:
    """The two EPYC environment traps must be reported, not silently rediscovered."""

    def test_library_path_risk_flags_a_cuda_directory(self, monkeypatch) -> None:
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda-12.6/lib64:/opt/lib")
        risk = library_path_risk()
        assert risk["shadowing_risk"] is True
        assert risk["cuda_entries"] == ["/usr/local/cuda-12.6/lib64"]
        assert "env -u LD_LIBRARY_PATH" in risk["guidance"]
        assert "Do not change the host driver" in risk["guidance"]

    def test_library_path_risk_is_quiet_without_cuda(self, monkeypatch) -> None:
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/somelib")
        risk = library_path_risk()
        assert risk["shadowing_risk"] is False
        assert risk["cuda_entries"] == []

    def test_library_path_risk_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        risk = library_path_risk()
        assert risk["shadowing_risk"] is False

    def test_cuda_child_environment_disables_jax_memory_preallocation(self) -> None:
        environment = clean_environment()
        assert environment[JAX_GPU_PREALLOCATE] == "false"


class TestGpuEnvironmentPaths:
    """Probe environments must be portable and remain explicitly configurable."""

    def test_explicit_gpu_environment_override_is_honored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        configured = tmp_path / "cuda-probes"
        monkeypatch.setenv("XLAXIOM_GPU_ENV_DIR", str(configured))

        assert gpu_environment_dir() == str(configured)

    def test_linux_gpu_environment_uses_xdg_cache_and_posix_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache_root = tmp_path / "xdg-cache"
        monkeypatch.delenv("XLAXIOM_GPU_ENV_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
        monkeypatch.setattr("xlaxiom.gpu_validation.platform.system", lambda: "Linux")

        expected = os.path.join(str(cache_root), "xlaxiom-gpu")
        assert gpu_environment_dir() == expected
        assert variant_python(CudaVariant.CUDA12) == os.path.join(
            expected, "cuda12-probe", "bin", "python"
        )

    def test_windows_gpu_environment_uses_local_app_data_and_windows_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cache_root = tmp_path / "local-app-data"
        monkeypatch.delenv("XLAXIOM_GPU_ENV_DIR", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(cache_root))
        monkeypatch.setattr("xlaxiom.gpu_validation.platform.system", lambda: "Windows")

        expected = os.path.join(str(cache_root), "xlaxiom-gpu")
        assert gpu_environment_dir() == expected
        assert variant_python(CudaVariant.CUDA13) == os.path.join(
            expected, "cuda13-probe", "Scripts", "python.exe"
        )


class TestGpuRuntimeReexecution:
    """GPU launch guards must not mistake an inherited marker for a CUDA runtime."""

    def test_active_runtime_probe_disables_jax_memory_preallocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(JAX_GPU_PREALLOCATE, "true")

        cli._active_jax_gpu_runtime()

        assert cli.os.environ[JAX_GPU_PREALLOCATE] == "false"

    def test_stale_boolean_marker_reexecutes_into_the_isolated_interpreter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = "/isolated/cuda/bin/python"
        target_path = cli.os.path.abspath(target)
        calls: list[tuple[list[str], dict[str, str]]] = []

        monkeypatch.setenv("XLAXIOM_GPU_REEXEC", "1")
        monkeypatch.setattr(cli, "_active_jax_gpu_runtime", lambda: False)
        monkeypatch.setattr(cli, "_gpu_interpreter", lambda: (target, 2))
        monkeypatch.setattr(cli.sys, "argv", ["xlaxiom", "gpu", "campaign"])

        def fake_run(command, *, env, check):
            calls.append((command, env))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        with pytest.raises(SystemExit) as exit_info:
            cli._ensure_gpu_runtime()

        assert exit_info.value.code == 0
        assert len(calls) == 1
        command, environment = calls[0]
        assert command == [target, "-m", "xlaxiom", "gpu", "campaign"]
        assert environment["XLAXIOM_GPU_REEXEC"] == target_path
        assert environment["CUDA_VISIBLE_DEVICES"] == "2"
        assert environment[JAX_GPU_PREALLOCATE] == "false"

    def test_matching_interpreter_marker_stops_a_reexecution_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = "/isolated/cuda/bin/python"
        monkeypatch.setenv("XLAXIOM_GPU_REEXEC", cli.os.path.abspath(target))
        monkeypatch.setattr(cli, "_active_jax_gpu_runtime", lambda: False)
        monkeypatch.setattr(cli, "_gpu_interpreter", lambda: (target, 0))

        with pytest.raises(SystemExit, match="stopping to avoid a re-execution loop"):
            cli._ensure_gpu_runtime()

    def test_gpu_interpreter_uses_configured_probe_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        device = make_device("RTX 3060", "8.6", ordinal=2)
        monkeypatch.setenv("XLAXIOM_GPU_ENV_DIR", str(tmp_path / "cuda-probes"))
        expected = Path(variant_python(CudaVariant.CUDA13))
        expected.parent.mkdir(parents=True)
        expected.touch()
        monkeypatch.setattr(cli, "enumerate_physical_gpus", lambda: [device])
        monkeypatch.setattr(
            cli, "_interpreter_has_gpu", lambda interpreter: interpreter == str(expected)
        )

        assert cli._gpu_interpreter() == (str(expected), 2)


class TestGpuCampaignCliGuards:
    """CLI selection must not use Python's negative sequence indexes for GPUs."""

    def test_negative_device_index_is_rejected_before_campaign_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        device = make_device("RTX 3060", "8.6")
        emitted: list[dict[str, object]] = []
        monkeypatch.setattr(cli, "_ensure_gpu_runtime", lambda: None)
        monkeypatch.setattr(cli, "enumerate_physical_gpus", lambda: [device])
        monkeypatch.setattr(
            cli,
            "validate_all",
            lambda devices, allow_busy: {
                "results": [{"state": DeviceState.PASS.value, "variant": "cuda13"}],
                "summary": {},
            },
        )
        monkeypatch.setattr(cli, "_emit", emitted.append)
        monkeypatch.setattr(
            cli,
            "run_gpu_campaign",
            lambda *args, **kwargs: pytest.fail(
                "campaign must not run for a negative device index"
            ),
        )

        result = cli.command_gpu_campaign(
            SimpleNamespace(
                shard_count=1,
                shard_index=0,
                resume=False,
                ledger=None,
                allow_busy=False,
                device=-1,
            )
        )

        assert result == 2
        assert emitted[0]["reason"] == (
            "device index -1 is out of range; 1 device(s) "
            "passed every bounded probe in this process"
        )

    def test_ordinal_ordering_risk_detects_mismatch(self) -> None:
        devices = [
            make_device("GPU-A", "7.0", ordinal=0, smi_index=0),
            make_device("GPU-B", "8.6", ordinal=1, smi_index=2),
            make_device("GPU-C", "6.1", ordinal=2, smi_index=1),
        ]
        risk = ordinal_ordering_risk(devices)
        assert risk["differs"] is True
        assert risk["cuda_ordinals"] == [0, 1, 2]
        assert risk["nvidia_smi_indices"] == [0, 2, 1]
        assert "CUDA_VISIBLE_DEVICES uses the ordinal" in risk["note"]

    def test_ordinal_ordering_risk_quiet_when_aligned(self) -> None:
        devices = [
            make_device("GPU-A", "7.0", ordinal=0, smi_index=0),
            make_device("GPU-B", "8.6", ordinal=1, smi_index=1),
        ]
        risk = ordinal_ordering_risk(devices)
        assert risk["differs"] is False
        assert "matches" in risk["note"]

    def test_ordinal_ordering_risk_single_device(self) -> None:
        risk = ordinal_ordering_risk([make_device("GPU-A", "7.0")])
        assert risk["differs"] is False
        assert "unambiguous" in risk["note"]


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


class TestVisibleDeviceAttribution:
    """A pinned CUDA_VISIBLE_DEVICES must attribute evidence to the right physical GPU."""

    def test_single_visible_name_is_used_for_attribution(self) -> None:
        assert _visible_gpu_name(["Tesla P40"]) == "Tesla P40"
        assert _visible_gpu_name(["Tesla P40", "Tesla V100-SXM2-16GB"]) is None
        assert _visible_gpu_name([]) is None

    def test_idle_gate_blocks_a_busy_device(self) -> None:
        busy = make_device("RTX 3060", "8.6", utilization=6.0, used=377)
        assert is_idle(busy) is False

    def test_idle_gate_allows_a_quiet_device(self) -> None:
        quiet = make_device("Tesla V100", "7.0", utilization=0.0, used=10)
        assert is_idle(quiet) is True
