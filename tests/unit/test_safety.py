from xlaxiom.generation import synthetic_control
from xlaxiom.pipeline import FaultInjection
from xlaxiom.safety import compare_isolated


def test_worker_timeout_terminates_child() -> None:
    result = compare_isolated(
        synthetic_control(), fault=FaultInjection(), timeout_s=0.1, _test_delay_s=1.0
    )
    assert result.status == "timeout"
