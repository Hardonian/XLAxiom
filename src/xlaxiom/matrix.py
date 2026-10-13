"""Isolated uv-backed execution for separately resolved JAX environments."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .regression import classify_versions
from .storage import load_case, save_json

_VERSION_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-+]*$")


def validate_jax_version(version: str) -> str:
    """Validate that a JAX version specifier is safe and free from argument injection."""
    if not isinstance(version, str) or not version or not _VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"Invalid or unsafe JAX version specifier: {version!r}")
    return version


@dataclass(frozen=True)
class MatrixEnvironment:
    name: str
    jax: str
    timeout_s: float = 90.0

    def __post_init__(self) -> None:
        validate_jax_version(self.jax)


def load_config(path: str | Path) -> tuple[MatrixEnvironment, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    environments = tuple(MatrixEnvironment(**item) for item in data["environments"])
    if not environments or len({env.name for env in environments}) != len(environments):
        raise ValueError("matrix requires uniquely named environments")
    return environments


def _run_environment(environment: MatrixEnvironment, case: Path) -> dict[str, Any]:
    validate_jax_version(environment.jax)
    command = [
        "uv",
        "run",
        "--isolated",
        "--with-editable",
        ".",
        "--with",
        f"jax=={environment.jax}",
        "python",
        "-m",
        "xlaxiom.matrix",
        "_worker",
        str(case),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=environment.timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "environment": asdict(environment),
            "worker_status": "timeout",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
    record: dict[str, Any] = {
        "environment": asdict(environment),
        "worker_status": "ok" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "duration_s": time.perf_counter() - started,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    try:
        record["comparison"] = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pass
    return record


def run_matrix(
    config_path: str | Path, case_path: str | Path, output: str | Path
) -> dict[str, Any]:
    case = Path(case_path).resolve()
    load_case(case)  # checksum and schema validation before creating environments
    results = [_run_environment(environment, case) for environment in load_config(config_path)]
    report = {"matrix_version": "1.0", "case": str(case.name), "results": results}
    if len(results) >= 2:
        report["pairwise"] = [
            {
                "older": results[index]["environment"]["name"],
                "newer": results[index + 1]["environment"]["name"],
                **classify_versions(results[index], results[index + 1]),
            }
            for index in range(len(results) - 1)
        ]
    save_json(output, report)
    return report


def _worker(case_path: str) -> int:
    from .pipeline import run_comparison

    case = load_case(case_path)
    result = run_comparison(case.workload, case.comparison_mode, case.fault)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv and len(argv) == 2 and argv[0] == "_worker":
        return _worker(argv[1])
    raise SystemExit("matrix module is invoked by xlaxiom matrix")


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
