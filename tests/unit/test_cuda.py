"""Unit tests for src/xlaxiom/cuda.py."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from xlaxiom.cuda import cuda_capability, driver_capability, recommended_variant
from xlaxiom.hardware import CudaVariant, PhysicalGpu


def _make_gpu(
    name: str = "NVIDIA GeForce RTX 4090",
    capability: str = "8.9",
    driver: str = "580.30",
    ordinal: int = 0,
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


class DummyDevice:
    """Mock JAX device with explicit __str__ representation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class TestCudaCapability:
    """Tests for cuda_capability()."""

    def test_jax_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = __import__

        def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("jax", "jaxlib"):
                raise ImportError("No module named 'jax'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", custom_import)

        result = cuda_capability()
        assert result["supported"] is False
        assert result["status"] == "unsupported"
        assert "No module named 'jax'" in result["reason"]

    def test_jax_devices_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_jax = MagicMock()
        mock_jaxlib = MagicMock()
        mock_jaxlib.__version__ = "0.5.0"
        mock_jax.devices.side_effect = RuntimeError("GPU backend initialization failed")

        monkeypatch.setitem(sys.modules, "jax", mock_jax)
        monkeypatch.setitem(sys.modules, "jaxlib", mock_jaxlib)

        result = cuda_capability()
        assert result["supported"] is False
        assert result["status"] == "unsupported"
        assert "RuntimeError" in result["reason"]
        assert result["jaxlib"] == "0.5.0"

    def test_no_jax_cuda_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_jax = MagicMock()
        mock_jaxlib = MagicMock()
        mock_jaxlib.__version__ = "0.5.0"
        mock_jax.devices.return_value = []

        monkeypatch.setitem(sys.modules, "jax", mock_jax)
        monkeypatch.setitem(sys.modules, "jaxlib", mock_jaxlib)

        result = cuda_capability()
        assert result["supported"] is False
        assert result["status"] == "unsupported"
        assert result["reason"] == "no JAX CUDA device"
        assert result["jaxlib"] == "0.5.0"

    def test_jax_cuda_devices_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_jax = MagicMock()
        mock_jaxlib = MagicMock()
        mock_jaxlib.__version__ = "0.5.0"
        mock_jax.devices.return_value = [DummyDevice("cuda:0"), DummyDevice("cuda:1")]

        monkeypatch.setitem(sys.modules, "jax", mock_jax)
        monkeypatch.setitem(sys.modules, "jaxlib", mock_jaxlib)

        result = cuda_capability()
        assert result["supported"] is True
        assert result["status"] == "available"
        assert result["jaxlib"] == "0.5.0"
        assert result["devices"] == ["cuda:0", "cuda:1"]


class TestDriverCapability:
    """Tests for driver_capability()."""

    def test_driver_capability_delegates_to_hardware_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected_report = {
            "environment": {"driver": "580.30"},
            "physical_gpus": [],
        }
        monkeypatch.setattr("xlaxiom.cuda.hardware_report", lambda: expected_report)

        result = driver_capability()
        assert result == expected_report


class TestRecommendedVariant:
    """Tests for recommended_variant()."""

    def test_no_physical_gpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("xlaxiom.cuda.enumerate_physical_gpus", lambda: [])

        result = recommended_variant()
        assert result["variant"] is None
        assert result["reason"] == "no NVIDIA GPU is visible to the driver"

    def test_cuda13_gpu_recommended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpu13 = _make_gpu(name="NVIDIA H100", capability="9.0", driver="580.30")
        monkeypatch.setattr("xlaxiom.cuda.enumerate_physical_gpus", lambda: [gpu13])
        monkeypatch.setattr(
            "xlaxiom.cuda.select_variant",
            lambda dev: (CudaVariant.CUDA13, "SM 9.0 >= SM 7.5 and driver 580.30 >= 580"),
        )

        result = recommended_variant()
        assert result["variant"] == "cuda13"
        assert result["device"] == "NVIDIA H100"
        assert "SM 9.0" in result["reason"]

    def test_cuda12_gpu_recommended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpu12 = _make_gpu(name="Tesla V100", capability="7.0", driver="525.60")
        monkeypatch.setattr("xlaxiom.cuda.enumerate_physical_gpus", lambda: [gpu12])

        def fake_select(dev: PhysicalGpu) -> tuple[CudaVariant | None, str]:
            if dev.name == "Tesla V100":
                return CudaVariant.CUDA12, "SM 7.0 >= SM 5.2 and driver 525.60 >= 525"
            return None, "unsupported"

        monkeypatch.setattr("xlaxiom.cuda.select_variant", fake_select)

        result = recommended_variant()
        assert result["variant"] == "cuda12"
        assert result["device"] == "Tesla V100"
        assert "SM 7.0" in result["reason"]

    def test_unsupported_gpu_variant_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gpu_old = _make_gpu(name="Tesla K80", capability="3.7", driver="470.00")
        monkeypatch.setattr("xlaxiom.cuda.enumerate_physical_gpus", lambda: [gpu_old])
        monkeypatch.setattr(
            "xlaxiom.cuda.select_variant",
            lambda dev: (None, "SM 3.7 < SM 5.2"),
        )

        result = recommended_variant()
        assert result["variant"] is None
        assert result["reason"] == "no visible GPU meets the CUDA 12 minimum of SM 5.2"
