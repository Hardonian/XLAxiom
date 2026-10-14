"""Integration coverage for JAX execution capability gating.

These tests pin the honest-failure contract of ``execute_jax``: a requested
platform that is not really present is reported as unsupported and never
silently substituted with CPU. They inject a stand-in ``jax`` module so the
device-availability branches are exercised without requiring physical
accelerator hardware.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xlaxiom.execution import (
    ExecutionStatus,
    _device_prefix,
    _platform_present,
    benchmark_jax,
    execute_jax,
    execute_numpy,
)
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.model import InputSpec, Node, Workload


def test_jax_execution_records_requested_cpu_device() -> None:
    result = execute_jax(generate(GenerationConfig(seed=2)), jit=True)
    assert result.status == ExecutionStatus.OK
    assert result.device is not None and "cpu" in result.device.lower()
    assert result.environment is not None
    assert result.environment["requested_platform"] == "cpu"


def test_generated_motifs_execute_with_numpy_and_jax() -> None:
    for seed in range(4):
        result = execute_jax(generate(GenerationConfig(seed=seed)), jit=True)
        assert result.status == ExecutionStatus.OK


def _workload() -> Workload:
    return generate(GenerationConfig(seed=3))


def test_device_prefix_maps_platform_to_device_name() -> None:
    # JAX names a CUDA device ``cuda:N`` while its platform attribute is ``gpu``.
    assert _device_prefix("gpu") == "cuda"
    assert _device_prefix("cpu") == "cpu"
    assert _device_prefix("tpu") == "tpu"
    assert _device_prefix("something-else") == "something-else"


def test_platform_present_matches_the_device_name_prefix() -> None:
    jax_stub = MagicMock()
    jax_stub.devices.return_value = ["cpu:0"]
    assert _platform_present(jax_stub, "cpu") is True
    assert _platform_present(jax_stub, "gpu") is False


@pytest.fixture
def fake_jax() -> types.ModuleType:
    """Install a controllable ``jax`` module for the duration of one test."""
    module = types.ModuleType("jax")
    submodule = types.ModuleType("jax.numpy")
    module.numpy = submodule
    with (
        patch.dict(sys.modules, {"jax": module, "jax.numpy": submodule}),
    ):
        yield module


def test_absent_accelerator_is_unsupported_and_never_falls_back_to_cpu(
    fake_jax: types.ModuleType,
) -> None:
    fake_jax.devices = lambda _platform=None: []
    fake_jax.__version__ = "0.0.0-test"

    result = execute_jax(_workload(), requested_platform="gpu")

    assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
    assert "no gpu device is available" in (result.error or "")
    assert "A CPU fallback was not performed." in (result.error or "")
    assert result.outputs == ()


def test_device_enumeration_failure_is_reported_as_unsupported(
    fake_jax: types.ModuleType,
) -> None:
    def explode(_platform: str | None = None) -> list[str]:
        raise RuntimeError("backend initialization failed")

    fake_jax.devices = explode
    fake_jax.__version__ = "0.0.0-test"

    result = execute_jax(_workload(), requested_platform="gpu")

    assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
    assert "backend initialization failed" in (result.error or "")


def test_out_of_range_device_index_is_rejected(fake_jax: types.ModuleType) -> None:
    device = MagicMock()
    device.__str__ = lambda _self: "cpu:0"
    fake_jax.devices = lambda _platform=None: [device]
    fake_jax.__version__ = "0.0.0-test"

    result = execute_jax(_workload(), requested_platform="cpu", device_index=5)

    assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
    assert "device index 5 out of range for 1 devices" in (result.error or "")


def test_missing_jax_import_is_unsupported() -> None:
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def deny_jax(name: str, *args: object, **kwargs: object) -> object:
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax is not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=deny_jax):
        result = execute_jax(_workload(), requested_platform="cpu")

    assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
    assert "jax is not installed" in (result.error or "")


def test_invalid_workload_is_classified_before_any_backend_runs() -> None:
    bad = Workload(
        identifier="forward-reference",
        seed=1,
        inputs=(InputSpec("x", (2, 2)),),
        nodes=(Node("neg", ("missing",), "out"),),
        outputs=("out",),
    )

    result = execute_jax(bad, requested_platform="cpu")

    assert result.status == ExecutionStatus.INVALID_WORKLOAD
    assert result.error


def test_numpy_execution_failure_is_classified_not_raised() -> None:
    bad = Workload(
        identifier="unknown-op",
        seed=1,
        inputs=(InputSpec("x", (2, 2)),),
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("not-an-output",),
    )

    result = execute_numpy(bad)

    assert result.status in {
        ExecutionStatus.EXECUTION_FAILURE,
        ExecutionStatus.INVALID_WORKLOAD,
    }
    assert result.error


def test_benchmark_jax_rejects_a_zero_measurement_budget() -> None:
    with pytest.raises(ValueError, match="measurements must be at least 1"):
        benchmark_jax(_workload(), measurements=0)


def test_benchmark_jax_raises_rather_than_benchmarking_on_cpu_for_absent_platform() -> None:
    """An absent accelerator must raise, never silently benchmark on CPU.

    ``_platform_present`` is stubbed so the accelerator branch is reached on a
    CPU-only host; the real ``jax`` module still imports normally.
    """
    with patch("xlaxiom.execution._platform_present", return_value=False):
        with pytest.raises(RuntimeError, match="A CPU fallback was not performed"):
            benchmark_jax(_workload(), measurements=3, requested_platform="gpu")


def test_execution_result_summary_reports_shape_dtype_and_finiteness() -> None:
    result = execute_numpy(_workload())

    summary = result.summary()

    assert summary["backend"] == "numpy"
    assert summary["status"] == "ok"
    assert summary["outputs"]
    first = summary["outputs"][0]
    assert isinstance(first["shape"], list)
    assert isinstance(first["dtype"], str)
    assert first["finite"] >= 0


def test_numpy_and_jax_agree_on_a_deterministic_workload() -> None:
    """Both backends agree within the recorded float32 tolerance.

    They are not bit-identical: float32 reduction order differs between NumPy
    and XLA, which is precisely the class of observation this tool exists to
    record. Exact equality would be a false claim.
    """
    workload = _workload()
    numpy_result = execute_numpy(workload)
    jax_result = execute_jax(workload, jit=True)

    assert numpy_result.status == ExecutionStatus.OK
    assert jax_result.status == ExecutionStatus.OK
    assert numpy_result.outputs[0].shape == jax_result.outputs[0].shape
    assert np.allclose(numpy_result.outputs[0], jax_result.outputs[0], rtol=1e-5, atol=1e-6)
