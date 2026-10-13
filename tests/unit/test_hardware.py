"""Unit tests for src/xlaxiom/hardware.py hardware discovery and capability gating."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from xlaxiom.hardware import (
    CudaVariant,
    PhysicalGpu,
    _cuda_runtime_names,
    _driver_cuda_version,
    _driver_version,
    _gpu_processes,
    _nvidia_smi_csv,
    _nvidia_smi_fields,
    driver_supported,
    enumerate_physical_gpus,
    environment_manifest,
    hardware_report,
    idle_assessment,
    is_idle,
    library_path_risk,
    ordinal_ordering_risk,
    select_variant,
    sm_supported,
)


class TestNvidiaSmiWrappers:
    def test_nvidia_smi_csv_no_executable(self) -> None:
        with patch("shutil.which", return_value=None):
            assert _nvidia_smi_csv("name") == []

    def test_nvidia_smi_csv_subprocess_error(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=subprocess.SubprocessError("Failed")),
        ):
            assert _nvidia_smi_csv("name") == []

    def test_nvidia_smi_csv_non_zero_returncode(self) -> None:
        mock_res = MagicMock(returncode=1)
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_res),
        ):
            assert _nvidia_smi_csv("name") == []

    def test_nvidia_smi_csv_success(self) -> None:
        mock_res = MagicMock(
            returncode=0,
            stdout="0, RTX 3080, 8.6\n1, RTX 3090, 8.6\n",
        )
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_res),
        ):
            rows = _nvidia_smi_csv("index,name,compute_cap")
            assert len(rows) == 2
            assert rows[0] == ["0", "RTX 3080", "8.6"]
            assert rows[1] == ["1", "RTX 3090", "8.6"]

    def test_nvidia_smi_fields_mismatched_row_length(self) -> None:
        with patch("xlaxiom.hardware._nvidia_smi_csv", return_value=[["0", "RTX 3080"]]):
            records = _nvidia_smi_fields("index,name,compute_cap")
            assert records == []

    def test_driver_version(self) -> None:
        mock_record = [{"driver_version": "535.100.00"}]
        with patch("xlaxiom.hardware._nvidia_smi_fields", return_value=mock_record):
            assert _driver_version() == "535.100.00"

        with patch("xlaxiom.hardware._nvidia_smi_fields", return_value=[]):
            assert _driver_version() == ""

    def test_driver_cuda_version(self) -> None:
        with patch("shutil.which", return_value=None):
            assert _driver_cuda_version() == ""

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=OSError("Exec error")),
        ):
            assert _driver_cuda_version() == ""

        mock_smi_out = (
            "+-----------------------------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 535.100.00             Driver Version: 535.100.00"
            "     CUDA Version: 12.2     |\n"
            "+-----------------------------------------------------------------------------------------+\n"
        )
        mock_res = MagicMock(returncode=0, stdout=mock_smi_out)
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_res),
        ):
            assert _driver_cuda_version() == "12.2"

    def test_gpu_processes(self) -> None:
        with patch("shutil.which", return_value=None):
            assert _gpu_processes() == {}

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=subprocess.SubprocessError("Failed")),
        ):
            assert _gpu_processes() == {}

        mock_res_err = MagicMock(returncode=1)
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_res_err),
        ):
            assert _gpu_processes() == {}

        mock_stdout = (
            "GPU-123, 1234, /usr/bin/python3, 500\n"
            "GPU-123, 5678, python script.py --arg, 1000\n"
            "GPU-456, 9999, , 200\n"
            "invalid line without enough fields\n"
        )
        mock_res = MagicMock(returncode=0, stdout=mock_stdout)
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_res),
        ):
            procs = _gpu_processes()
            assert "GPU-123" in procs
            assert procs["GPU-123"] == ("pid1234:python3", "pid5678:python")
            assert "GPU-456" in procs
            assert procs["GPU-456"] == ("pid9999:unknown",)


class TestCudaRuntimeNamesAndEnumeration:
    def test_cuda_runtime_names_no_libraries(self) -> None:
        with patch("glob.glob", return_value=[]):
            assert _cuda_runtime_names() == []

    def test_cuda_runtime_names_ctypes_error(self) -> None:
        with (
            patch("glob.glob", return_value=["/path/to/libcudart.so"]),
            patch("ctypes.CDLL", side_effect=OSError("Cannot open lib")),
        ):
            assert _cuda_runtime_names() == []

    def test_cuda_runtime_names_success(self) -> None:
        mock_dll = MagicMock()

        def mock_get_count(ref):
            ref._obj.value = 1
            return 0

        mock_dll.cudaGetDeviceCount.side_effect = mock_get_count

        def mock_get_props(buf, index):
            buf.raw = b"NVIDIA RTX 3090" + b"\x00" * 200
            return 0

        mock_dll.cudaGetDeviceProperties.side_effect = mock_get_props

        with (
            patch("glob.glob", return_value=["/path/to/libcudart.so"]),
            patch("ctypes.CDLL", return_value=mock_dll),
        ):
            names = _cuda_runtime_names()
            assert names == ["NVIDIA RTX 3090"]

    def test_enumerate_physical_gpus_no_records(self) -> None:
        with patch("xlaxiom.hardware._nvidia_smi_fields", return_value=[]):
            assert enumerate_physical_gpus() == []

    def test_enumerate_physical_gpus_corrupted_field(self) -> None:
        bad_records = [
            {
                "index": "not-an-int",
                "name": "GPU 0",
                "uuid": "GPU-123",
                "compute_cap": "8.6",
                "memory.total": "1000",
                "memory.used": "100",
                "memory.free": "900",
                "utilization.gpu": "0",
            }
        ]

        def mock_smi_fields(fields: str) -> list[dict[str, str]]:
            if "driver_version" in fields:
                return [{"driver_version": "535.100.00"}]
            return bad_records

        with patch("xlaxiom.hardware._nvidia_smi_fields", side_effect=mock_smi_fields):
            assert enumerate_physical_gpus() == []

    def test_enumerate_physical_gpus_single_visible_gpu(self) -> None:
        records = [
            {
                "index": "0",
                "name": "NVIDIA RTX 3080",
                "uuid": "GPU-3080",
                "compute_cap": "8.6",
                "memory.total": "10000",
                "memory.used": "1000",
                "memory.free": "9000",
                "utilization.gpu": "0.0",
            },
            {
                "index": "1",
                "name": "NVIDIA RTX 3090",
                "uuid": "GPU-3090",
                "compute_cap": "8.6",
                "memory.total": "24000",
                "memory.used": "2000",
                "memory.free": "22000",
                "utilization.gpu": "0.0",
            },
        ]

        def mock_smi_fields(fields: str) -> list[dict[str, str]]:
            if "driver_version" in fields:
                return [{"driver_version": "535.100.00"}]
            return records

        with (
            patch("xlaxiom.hardware._nvidia_smi_fields", side_effect=mock_smi_fields),
            patch("xlaxiom.hardware._driver_version", return_value="535.100.00"),
            patch("xlaxiom.hardware._driver_cuda_version", return_value="12.2"),
            patch("xlaxiom.hardware._gpu_processes", return_value={}),
            patch("xlaxiom.hardware._cuda_runtime_names", return_value=["NVIDIA RTX 3090"]),
        ):
            gpus = enumerate_physical_gpus()
            assert len(gpus) == 1
            assert gpus[0].name == "NVIDIA RTX 3090"
            assert gpus[0].ordinal == 0
            assert gpus[0].smi_index == 1

    def test_enumerate_physical_gpus_matching_cuda_runtime_order(self) -> None:
        records = [
            {
                "index": "0",
                "name": "NVIDIA RTX 3080",
                "uuid": "GPU-3080",
                "compute_cap": "8.6",
                "memory.total": "10000",
                "memory.used": "1000",
                "memory.free": "9000",
                "utilization.gpu": "0.0",
            },
            {
                "index": "1",
                "name": "NVIDIA RTX 3090",
                "uuid": "GPU-3090",
                "compute_cap": "8.6",
                "memory.total": "24000",
                "memory.used": "2000",
                "memory.free": "22000",
                "utilization.gpu": "0.0",
            },
        ]

        def mock_smi_fields(fields: str) -> list[dict[str, str]]:
            if "driver_version" in fields:
                return [{"driver_version": "535.100.00"}]
            return records

        mock_rt_names = ["NVIDIA RTX 3090", "NVIDIA RTX 3080"]
        with (
            patch("xlaxiom.hardware._nvidia_smi_fields", side_effect=mock_smi_fields),
            patch("xlaxiom.hardware._driver_version", return_value="535.100.00"),
            patch("xlaxiom.hardware._driver_cuda_version", return_value="12.2"),
            patch("xlaxiom.hardware._gpu_processes", return_value={}),
            patch("xlaxiom.hardware._cuda_runtime_names", return_value=mock_rt_names),
        ):
            gpus = enumerate_physical_gpus()
            assert len(gpus) == 2
            assert gpus[0].name == "NVIDIA RTX 3090"
            assert gpus[0].ordinal == 0
            assert gpus[1].name == "NVIDIA RTX 3080"
            assert gpus[1].ordinal == 1

    def test_enumerate_physical_gpus_fallback_order(self) -> None:
        records = [
            {
                "index": "0",
                "name": "NVIDIA RTX 3080",
                "uuid": "GPU-3080",
                "compute_cap": "8.6",
                "memory.total": "10000",
                "memory.used": "1000",
                "memory.free": "9000",
                "utilization.gpu": "0.0",
            }
        ]

        def mock_smi_fields(fields: str) -> list[dict[str, str]]:
            if "driver_version" in fields:
                return [{"driver_version": "535.100.00"}]
            return records

        with (
            patch("xlaxiom.hardware._nvidia_smi_fields", side_effect=mock_smi_fields),
            patch("xlaxiom.hardware._driver_version", return_value="535.100.00"),
            patch("xlaxiom.hardware._driver_cuda_version", return_value="12.2"),
            patch("xlaxiom.hardware._gpu_processes", return_value={}),
            patch("xlaxiom.hardware._cuda_runtime_names", return_value=[]),
        ):
            gpus = enumerate_physical_gpus()
            assert len(gpus) == 1
            assert gpus[0].name == "NVIDIA RTX 3080"
            assert gpus[0].ordinal == 0


class TestGatingRiskAndReporting:
    def test_sm_and_driver_supported(self) -> None:
        assert sm_supported((5, 2), CudaVariant.CUDA12) is True
        assert sm_supported((5, 1), CudaVariant.CUDA12) is False
        assert sm_supported((7, 5), CudaVariant.CUDA13) is True
        assert sm_supported((7, 4), CudaVariant.CUDA13) is False

        assert driver_supported("525.0.0", CudaVariant.CUDA12) is True
        assert driver_supported("524.0.0", CudaVariant.CUDA12) is False
        assert driver_supported("580.0.0", CudaVariant.CUDA13) is True
        assert driver_supported("579.0.0", CudaVariant.CUDA13) is False
        assert driver_supported("invalid", CudaVariant.CUDA12) is False
        assert driver_supported("", CudaVariant.CUDA12) is False

    def test_select_variant(self) -> None:
        # CUDA 13 eligible
        gpu13 = PhysicalGpu(
            ordinal=0,
            name="A100",
            uuid_redacted="dev-1",
            compute_capability="8.0",
            sm=(8, 0),
            driver_version="580.10",
            cuda_version="13.0",
            total_vram_mib=40000,
            used_vram_mib=0,
            free_vram_mib=40000,
            utilization_percent=0.0,
        )
        variant, reason = select_variant(gpu13)
        assert variant == CudaVariant.CUDA13
        assert ">= 580" in reason

        # CUDA 12 eligible (due to driver)
        gpu12 = PhysicalGpu(
            ordinal=0,
            name="A100",
            uuid_redacted="dev-1",
            compute_capability="8.0",
            sm=(8, 0),
            driver_version="535.10",
            cuda_version="12.2",
            total_vram_mib=40000,
            used_vram_mib=0,
            free_vram_mib=40000,
            utilization_percent=0.0,
        )
        variant, reason = select_variant(gpu12)
        assert variant == CudaVariant.CUDA12
        assert "driver 535.10 < 580" in reason

        # CUDA 12 eligible (due to SM)
        gpu_sm70 = PhysicalGpu(
            ordinal=0,
            name="V100",
            uuid_redacted="dev-2",
            compute_capability="7.0",
            sm=(7, 0),
            driver_version="580.10",
            cuda_version="13.0",
            total_vram_mib=16000,
            used_vram_mib=0,
            free_vram_mib=16000,
            utilization_percent=0.0,
        )
        variant, reason = select_variant(gpu_sm70)
        assert variant == CudaVariant.CUDA12
        assert "CUDA 13 requires SM >= 7.5" in reason

        # Unsupported
        gpu_old = PhysicalGpu(
            ordinal=0,
            name="K80",
            uuid_redacted="dev-3",
            compute_capability="3.7",
            sm=(3, 7),
            driver_version="535.10",
            cuda_version="12.0",
            total_vram_mib=12000,
            used_vram_mib=0,
            free_vram_mib=12000,
            utilization_percent=0.0,
        )
        variant, reason = select_variant(gpu_old)
        assert variant is None
        assert "below the CUDA 12 minimum" in reason

    def test_is_idle_and_idle_assessment(self) -> None:
        idle_gpu = PhysicalGpu(
            ordinal=0,
            name="RTX 3090",
            uuid_redacted="dev-1",
            compute_capability="8.6",
            sm=(8, 6),
            driver_version="535.10",
            cuda_version="12.0",
            total_vram_mib=24000,
            used_vram_mib=100,
            free_vram_mib=23900,
            utilization_percent=0.0,
            active_processes=(),
        )
        assert is_idle(idle_gpu) is True
        assessment = idle_assessment(idle_gpu)
        assert assessment["external_workload"] is False
        assert assessment["used_vram_mib"] == 100

        busy_gpu = PhysicalGpu(
            ordinal=0,
            name="RTX 3090",
            uuid_redacted="dev-1",
            compute_capability="8.6",
            sm=(8, 6),
            driver_version="535.10",
            cuda_version="12.0",
            total_vram_mib=24000,
            used_vram_mib=1000,
            free_vram_mib=23000,
            utilization_percent=10.0,
            active_processes=("pid123:python",),
        )
        assert is_idle(busy_gpu) is False

    def test_library_path_risk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/usr/lib")
        risk = library_path_risk()
        assert risk["shadowing_risk"] is True
        assert "/usr/local/cuda/lib64" in risk["cuda_entries"]

        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib:/usr/local/lib")
        risk = library_path_risk()
        assert risk["shadowing_risk"] is False

    def test_ordinal_ordering_risk(self) -> None:
        gpu1 = PhysicalGpu(
            ordinal=0,
            name="GPU 0",
            uuid_redacted="dev-1",
            compute_capability="8.6",
            sm=(8, 6),
            driver_version="535.10",
            cuda_version="12.0",
            total_vram_mib=24000,
            used_vram_mib=0,
            free_vram_mib=24000,
            utilization_percent=0.0,
            smi_index=0,
        )
        gpu2 = PhysicalGpu(
            ordinal=1,
            name="GPU 1",
            uuid_redacted="dev-2",
            compute_capability="8.6",
            sm=(8, 6),
            driver_version="535.10",
            cuda_version="12.0",
            total_vram_mib=24000,
            used_vram_mib=0,
            free_vram_mib=24000,
            utilization_percent=0.0,
            smi_index=1,
        )
        assert ordinal_ordering_risk([gpu1])["differs"] is False
        assert ordinal_ordering_risk([gpu1, gpu2])["differs"] is False

        gpu2_swapped = PhysicalGpu(
            ordinal=1,
            name="GPU 1",
            uuid_redacted="dev-2",
            compute_capability="8.6",
            sm=(8, 6),
            driver_version="535.10",
            cuda_version="12.0",
            total_vram_mib=24000,
            used_vram_mib=0,
            free_vram_mib=24000,
            utilization_percent=0.0,
            smi_index=0,
        )
        assert ordinal_ordering_risk([gpu1, gpu2_swapped])["differs"] is True

    def test_environment_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
        manifest = environment_manifest()
        assert manifest["os"]
        assert manifest["ld_library_path_present"] is True
        assert "ld_library_path_risk" in manifest

    def test_hardware_report(self) -> None:
        with patch("xlaxiom.hardware.enumerate_physical_gpus", return_value=[]):
            report = hardware_report()
            assert report["report"] == "XLAxiom CUDA hardware inventory"
            assert report["device_count"] == 0
            assert report["devices"] == []
