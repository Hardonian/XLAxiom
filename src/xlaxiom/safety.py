"""Spawned worker boundary for bounded, crash-contained comparisons."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any

from .model import Workload
from .pipeline import FaultInjection, run_comparison


@dataclass(frozen=True)
class WorkerResult:
    status: str
    duration_ms: float
    record: dict[str, Any] | None = None
    error: str | None = None


def _comparison_worker(
    payload: dict[str, Any], mode: str, fault_data: dict[str, Any] | None, delay_s: float, output: Any
) -> None:
    """Top-level target required by Windows' spawn multiprocessing implementation."""
    try:
        if delay_s:
            time.sleep(delay_s)
        workload = Workload.from_dict(payload)
        fault = FaultInjection(**fault_data) if fault_data else None
        output.put({"kind": "result", "comparison": run_comparison(workload, mode, fault).to_dict()})
    except BaseException:
        output.put({"kind": "crash", "error": traceback.format_exc()})


def compare_isolated(
    workload: Workload,
    mode: str = "jax-jit",
    fault: FaultInjection | None = None,
    timeout_s: float = 30.0,
    _test_delay_s: float = 0.0,
) -> WorkerResult:
    """Run one declared case in a spawned child, terminating it on timeout.

    The delay parameter exists solely for deterministic timeout tests and is never read from an artifact.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    context = mp.get_context("spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(
        target=_comparison_worker,
        args=(
            workload.canonical_dict(),
            mode,
            {"output_offset": fault.output_offset, "label": fault.label} if fault else None,
            _test_delay_s,
            output,
        ),
    )
    started = time.perf_counter()
    process.start()
    process.join(timeout_s)
    duration_ms = (time.perf_counter() - started) * 1000
    if process.is_alive():
        process.terminate()
        process.join(5)
        output.close()
        return WorkerResult("timeout", duration_ms, error=f"worker exceeded {timeout_s}s")
    try:
        message = output.get(timeout=1)
    except queue.Empty:
        output.close()
        return WorkerResult(
            "crash", duration_ms, error=f"worker exited {process.exitcode} without an observation"
        )
    output.close()
    if message["kind"] == "crash":
        return WorkerResult("crash", duration_ms, error=str(message["error"]))
    return WorkerResult("ok", duration_ms, record=dict(message["comparison"]))
