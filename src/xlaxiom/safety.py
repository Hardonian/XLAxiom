"""Spawned worker boundary for bounded, crash-contained comparisons."""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
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
    memory_peak_bytes: int | None = None
    termination: str | None = None


def _comparison_worker(
    payload: dict[str, Any],
    mode: str,
    fault_data: dict[str, Any] | None,
    delay_s: float,
    output: Any,
) -> None:
    """Top-level target required by Windows' spawn multiprocessing implementation."""
    try:
        if delay_s:
            time.sleep(delay_s)
        workload = Workload.from_dict(payload)
        fault = FaultInjection(**fault_data) if fault_data else None
        output.put(
            {"kind": "result", "comparison": run_comparison(workload, mode, fault).to_dict()}
        )
    except BaseException:
        output.put({"kind": "crash", "error": traceback.format_exc()})


def _resident_memory_bytes(pid: int | None) -> int | None:
    """Return a worker's resident memory without adding a monitoring dependency.

    Linux exposes RSS through ``/proc`` and Windows exposes the same information via
    ``GetProcessMemoryInfo``.  Other platforms report an unavailable sample; callers
    retain their timeout containment and record that memory could not be sampled.
    """
    if pid is None:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return None
            process = windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
            if not process:
                return None
            try:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if not windll.psapi.GetProcessMemoryInfo(
                    process, ctypes.byref(counters), counters.cb
                ):
                    return None
                return int(counters.WorkingSetSize)
            finally:
                windll.kernel32.CloseHandle(process)
        except (AttributeError, OSError):
            return None
    if os.path.exists(f"/proc/{pid}/status"):
        try:
            for line in open(f"/proc/{pid}/status", encoding="utf-8"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    if platform.system() == "Darwin":
        try:
            if pid == os.getpid():
                import resource

                rusage_self = getattr(resource, "RUSAGE_SELF", 0)
                getrusage = getattr(resource, "getrusage", None)
                if callable(getrusage):
                    return int(getrusage(rusage_self).ru_maxrss)
            import subprocess

            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL
            )
            return int(out.strip()) * 1024
        except Exception:
            return None
    return None


def _terminate_worker(process: Any) -> str:
    """Stop a contained worker and escalate only if normal termination failed."""
    process.terminate()
    process.join(5)
    if not process.is_alive():
        return "terminated"
    kill = getattr(process, "kill", None)
    if callable(kill):
        kill()
        process.join(5)
        return "killed" if not process.is_alive() else "kill_failed"
    return "termination_failed"


def _close_queue(output: Any) -> None:
    """Release queue handles after the worker has exited on every platform."""
    output.close()
    try:
        output.join_thread()
    except (AttributeError, OSError):
        pass


def compare_isolated(
    workload: Workload,
    mode: str = "jax-jit",
    fault: FaultInjection | None = None,
    timeout_s: float = 30.0,
    memory_limit_mb: int | None = None,
    _test_delay_s: float = 0.0,
) -> WorkerResult:
    """Run one declared case in a spawned child with time and memory containment.

    The delay parameter exists solely for deterministic timeout tests and is never read from
    an artifact.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if memory_limit_mb is not None and memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive when supplied")
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
    deadline = started + timeout_s
    memory_limit_bytes = memory_limit_mb * 1024 * 1024 if memory_limit_mb is not None else None
    peak_memory: int | None = None
    limit_exceeded = False
    while process.is_alive():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        process.join(min(0.05, remaining))
        if memory_limit_bytes is None or not process.is_alive():
            continue
        resident = _resident_memory_bytes(process.pid)
        if resident is not None:
            peak_memory = max(peak_memory or 0, resident)
            if resident > memory_limit_bytes:
                limit_exceeded = True
                break
    duration_ms = (time.perf_counter() - started) * 1000
    if process.is_alive():
        termination = _terminate_worker(process)
        _close_queue(output)
        if limit_exceeded:
            return WorkerResult(
                "memory_limit",
                duration_ms,
                error=f"worker exceeded {memory_limit_mb} MiB resident-memory limit",
                memory_peak_bytes=peak_memory,
                termination=termination,
            )
        return WorkerResult(
            "timeout",
            duration_ms,
            error=f"worker exceeded {timeout_s}s",
            memory_peak_bytes=peak_memory,
            termination=termination,
        )
    try:
        message = output.get(timeout=1)
    except queue.Empty:
        _close_queue(output)
        return WorkerResult(
            "crash",
            duration_ms,
            error=f"worker exited {process.exitcode} without an observation",
            memory_peak_bytes=peak_memory,
        )
    _close_queue(output)
    if message["kind"] == "crash":
        return WorkerResult(
            "crash", duration_ms, error=str(message["error"]), memory_peak_bytes=peak_memory
        )
    return WorkerResult(
        "ok", duration_ms, record=dict(message["comparison"]), memory_peak_bytes=peak_memory
    )
