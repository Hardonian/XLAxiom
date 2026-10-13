import os

from xlaxiom.generation import synthetic_control
from xlaxiom.pipeline import FaultInjection
from xlaxiom.safety import _resident_memory_bytes, compare_isolated


def test_current_process_memory_is_sampled_on_supported_hosts() -> None:
    sample = _resident_memory_bytes(os.getpid())
    if os.name in {"nt", "posix"}:
        assert sample is not None and sample > 0


def test_worker_timeout_terminates_child() -> None:
    result = compare_isolated(
        synthetic_control(), fault=FaultInjection(), timeout_s=0.1, _test_delay_s=1.0
    )
    assert result.status == "timeout"
    assert result.termination in {"terminated", "killed"}


def test_worker_memory_limit_containment() -> None:
    result = compare_isolated(
        synthetic_control(),
        fault=FaultInjection(),
        timeout_s=10.0,
        memory_limit_mb=1,
        _test_delay_s=1.0,
    )
    assert result.status == "memory_limit"
    assert result.memory_peak_bytes is not None
    assert result.termination in {"terminated", "killed"}
