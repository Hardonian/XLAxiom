"""Unit coverage for the spawned worker boundary in safety.py.

The worker boundary is XLAxiom's core containment claim: a case runs in a separate
interpreter with a time and memory budget, and every exit path is classified honestly.
These tests exercise the exit paths that do not require real GPU or OOM conditions.
"""

from __future__ import annotations

import os
import queue
from unittest.mock import MagicMock, patch

import pytest

from xlaxiom.generation import synthetic_control
from xlaxiom.pipeline import FaultInjection
from xlaxiom.safety import (
    WorkerResult,
    _close_queue,
    _comparison_worker,
    _resident_memory_bytes,
    _terminate_worker,
    compare_isolated,
)


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


def test_worker_result_defaults_are_none() -> None:
    result = WorkerResult(status="ok", duration_ms=1.0)
    assert result.record is None
    assert result.error is None
    assert result.memory_peak_bytes is None
    assert result.termination is None


def test_resident_memory_returns_none_for_missing_pid() -> None:
    assert _resident_memory_bytes(None) is None


def test_resident_memory_returns_none_for_dead_pid() -> None:
    # A pid that cannot exist keeps the sampler honest instead of raising.
    assert _resident_memory_bytes(-1) is None


def test_comparison_worker_emits_a_result_record() -> None:
    output: MagicMock = MagicMock()
    workload = synthetic_control()
    _comparison_worker(workload.canonical_dict(), "jax-jit", None, 0.0, output)

    message = output.put.call_args.args[0]
    assert message["kind"] == "result"
    assert "comparison" in message
    assert message["comparison"]["classification"]


def test_comparison_worker_emits_a_crash_record_for_bad_payload() -> None:
    output: MagicMock = MagicMock()
    _comparison_worker({"identifier": "not-a-real-workload"}, "jax-jit", None, 0.0, output)

    message = output.put.call_args.args[0]
    assert message["kind"] == "crash"
    assert "error" in message


def test_comparison_worker_passes_fault_injection_through() -> None:
    output: MagicMock = MagicMock()
    workload = synthetic_control()
    fault = {"output_offset": 0.5, "label": "test-fault"}
    _comparison_worker(workload.canonical_dict(), "jax-jit", fault, 0.0, output)

    message = output.put.call_args.args[0]
    assert message["kind"] == "result"
    record = message["comparison"]
    assert record["classification"] != "pass"


def test_comparison_worker_rejects_unknown_mode() -> None:
    output: MagicMock = MagicMock()
    workload = synthetic_control()
    _comparison_worker(workload.canonical_dict(), "not-a-backend", None, 0.0, output)

    message = output.put.call_args.args[0]
    assert message["kind"] == "crash"


def test_terminate_worker_terminates_a_live_process() -> None:
    process = MagicMock()
    process.is_alive.return_value = False  # gone after terminate + join
    assert _terminate_worker(process) == "terminated"
    process.terminate.assert_called_once()


def test_terminate_worker_escalates_to_kill() -> None:
    process = MagicMock()
    process.is_alive.side_effect = [True, False]  # survives terminate, dies to kill
    assert _terminate_worker(process) == "killed"
    process.kill.assert_called_once()


def test_terminate_worker_reports_kill_failure() -> None:
    process = MagicMock()
    process.is_alive.return_value = True
    assert _terminate_worker(process) == "kill_failed"


def test_terminate_worker_reports_failure_without_kill_support() -> None:
    process = MagicMock(spec=["terminate", "join", "is_alive"])
    process.is_alive.return_value = True
    assert _terminate_worker(process) == "termination_failed"


def test_close_queue_tolerates_missing_join_thread() -> None:
    output = MagicMock()
    output.join_thread.side_effect = AttributeError("no thread")
    _close_queue(output)  # must not raise
    output.close.assert_called_once()


def test_close_queue_tolerates_oserror_from_join_thread() -> None:
    output = MagicMock()
    output.join_thread.side_effect = OSError("closed")
    _close_queue(output)
    output.close.assert_called_once()


def test_compare_isolated_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        compare_isolated(synthetic_control(), timeout_s=0.0)


def test_compare_isolated_rejects_non_positive_memory_limit() -> None:
    with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
        compare_isolated(synthetic_control(), memory_limit_mb=0)


def test_compare_isolated_reports_crash_when_worker_dies_silently() -> None:
    with patch("xlaxiom.safety.mp.get_context") as context:
        process = MagicMock()
        # Not alive after start, and the queue yields nothing.
        process.is_alive.return_value = False
        process.exitcode = 1
        output = MagicMock()
        output.get.side_effect = queue.Empty
        context.return_value.Queue.return_value = output
        context.return_value.Process.return_value = process

        result = compare_isolated(synthetic_control(), timeout_s=5.0)

    assert result.status == "crash"
    assert "without an observation" in (result.error or "")


def test_compare_isolated_reports_crash_from_worker_payload() -> None:
    with patch("xlaxiom.safety.mp.get_context") as context:
        process = MagicMock()
        process.is_alive.return_value = False
        output = MagicMock()
        output.get.return_value = {"kind": "crash", "error": "ValueError: bad workload"}
        context.return_value.Queue.return_value = output
        context.return_value.Process.return_value = process

        result = compare_isolated(synthetic_control(), timeout_s=5.0)

    assert result.status == "crash"
    assert "bad workload" in (result.error or "")


def test_compare_isolated_returns_ok_for_a_successful_worker() -> None:
    with patch("xlaxiom.safety.mp.get_context") as context:
        process = MagicMock()
        process.is_alive.return_value = False
        output = MagicMock()
        output.get.return_value = {
            "kind": "result",
            "comparison": {"classification": "pass", "details": {}},
        }
        context.return_value.Queue.return_value = output
        context.return_value.Process.return_value = process

        result = compare_isolated(synthetic_control(), timeout_s=5.0)

    assert result.status == "ok"
    assert result.record == {"classification": "pass", "details": {}}
    assert result.error is None
    assert result.termination is None


def test_compare_isolated_records_memory_peak_when_sampled() -> None:
    with (
        patch("xlaxiom.safety.mp.get_context") as context,
        patch("xlaxiom.safety._resident_memory_bytes", return_value=64 * 1024 * 1024),
    ):
        process = MagicMock()
        # Guard alive, sampler probe alive (so it samples), guard dead to exit the
        # loop, and the post-loop liveness check dead.
        process.is_alive.side_effect = [True, True, False, False]
        output = MagicMock()
        output.get.return_value = {
            "kind": "result",
            "comparison": {"classification": "pass"},
        }
        context.return_value.Queue.return_value = output
        context.return_value.Process.return_value = process

        result = compare_isolated(synthetic_control(), timeout_s=5.0, memory_limit_mb=512)

    assert result.status == "ok"
    assert result.memory_peak_bytes == 64 * 1024 * 1024


def test_compare_isolated_timeout_path_terminates_the_worker() -> None:
    with patch("xlaxiom.safety.mp.get_context") as context:
        process = MagicMock()
        process.is_alive.return_value = True  # never exits
        context.return_value.Queue.return_value = MagicMock()
        context.return_value.Process.return_value = process

        result = compare_isolated(synthetic_control(), timeout_s=0.05)

    assert result.status == "timeout"
    assert "exceeded" in (result.error or "")
    assert result.termination in {"terminated", "killed", "kill_failed", "termination_failed"}


def test_compare_isolated_memory_limit_path_is_distinguished_from_timeout() -> None:
    with (
        patch("xlaxiom.safety.mp.get_context") as context,
        patch("xlaxiom.safety._resident_memory_bytes", return_value=4 * 1024 * 1024 * 1024),
    ):
        process = MagicMock()
        process.is_alive.return_value = True
        context.return_value.Queue.return_value = MagicMock()
        context.return_value.Process.return_value = process

        result = compare_isolated(synthetic_control(), timeout_s=30.0, memory_limit_mb=64)

    assert result.status == "memory_limit"
    assert "MiB resident-memory limit" in (result.error or "")
    assert result.memory_peak_bytes == 4 * 1024 * 1024 * 1024
