import pytest

from xlaxiom.comparison import Classification
from xlaxiom.execution import ExecutionStatus
from xlaxiom.generation import synthetic_control
from xlaxiom.pipeline import (
    ExecutionTarget,
    FaultInjection,
    _kwargs,
    run_backend,
    run_comparison,
)


def test_fault_injection_and_execution_target_defaults() -> None:
    fault = FaultInjection()
    assert fault.output_offset == 0.0
    assert fault.label == "synthetic-control"

    custom_fault = FaultInjection(output_offset=0.5, label="custom")
    assert custom_fault.output_offset == 0.5
    assert custom_fault.label == "custom"

    target = ExecutionTarget()
    assert target.platform == "cpu"
    assert target.device_index is None

    custom_target = ExecutionTarget(platform="gpu", device_index=1)
    assert custom_target.platform == "gpu"
    assert custom_target.device_index == 1


def test_kwargs_helper() -> None:
    assert _kwargs(None) == {"requested_platform": "cpu"}
    target = ExecutionTarget(platform="cuda", device_index=0)
    assert _kwargs(target) == {"requested_platform": "cuda", "device_index": 0}


def test_run_comparison_modes() -> None:
    workload = synthetic_control(7)

    res_jit = run_comparison(workload, mode="jax-jit")
    assert res_jit.classification == Classification.PASS

    res_eager = run_comparison(workload, mode="jax-eager")
    assert res_eager.classification == Classification.PASS

    res_eager_jit = run_comparison(workload, mode="jax-eager-vs-jit")
    assert res_eager_jit.classification == Classification.PASS


def test_run_comparison_with_fault_injection() -> None:
    workload = synthetic_control(7)
    fault = FaultInjection(output_offset=0.125)
    res = run_comparison(workload, mode="jax-jit", fault=fault)
    assert res.classification == Classification.NUMERICAL_DISCREPANCY


def test_run_comparison_invalid_mode() -> None:
    workload = synthetic_control(7)
    with pytest.raises(ValueError, match="unsupported comparison mode invalid-mode"):
        run_comparison(workload, mode="invalid-mode")


def test_run_backend() -> None:
    workload = synthetic_control(7)

    res_numpy = run_backend(workload, backend="numpy")
    assert res_numpy.status == ExecutionStatus.OK
    assert res_numpy.backend == "numpy"

    res_eager = run_backend(workload, backend="jax-eager")
    assert res_eager.status == ExecutionStatus.OK
    assert res_eager.backend == "jax-cpu"

    res_jit = run_backend(workload, backend="jax-jit")
    assert res_jit.status == ExecutionStatus.OK
    assert res_jit.backend == "jax-cpu"

    with pytest.raises(ValueError, match="unsupported backend invalid-backend"):
        run_backend(workload, backend="invalid-backend")


def test_run_comparison_and_backend_with_target() -> None:
    workload = synthetic_control(7)
    target = ExecutionTarget(platform="nonexistent_device", device_index=0)

    # When requesting an unavailable device platform, execution status should reflect backend issue
    res_comp = run_comparison(workload, mode="jax-jit", target=target)
    assert res_comp.classification == Classification.UNSUPPORTED_BACKEND

    res_backend = run_backend(workload, backend="jax-jit", target=target)
    assert res_backend.status == ExecutionStatus.UNSUPPORTED_BACKEND
