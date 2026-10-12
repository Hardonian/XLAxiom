"""Isolated version runner for the trusted #40634 study adapter."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .storage import save_json


def run_hyperbolic_matrix(versions: list[str], output: str) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for version in versions:
        started = time.perf_counter()
        command = [
            "uv",
            "run",
            "--isolated",
            "--with-editable",
            ".",
            "--with",
            f"jax=={version}",
            "python",
            "-m",
            "xlaxiom.historical",
            "_study",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        result: dict[str, Any] = {
            "requested_jax": version,
            "exit_code": completed.returncode,
            "duration_s": time.perf_counter() - started,
            "stderr": completed.stderr[-2000:],
        }
        if completed.returncode == 0:
            result["study"] = json.loads(completed.stdout.splitlines()[-1])
        results.append(result)
    report = {
        "report_version": "1.0",
        "results": results,
        "fix_release_status": "PENDING: upstream commit 86d3fe7 is not verified in a released wheel.",
    }
    save_json(output, report)
    return report
