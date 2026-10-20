"""Generate zero-cost, explicitly simulated TPU qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evidence" / "tpu-offline"
ARTIFACT_NAMES = (
    "cpu-jit-evidence.json",
    "simulated-tpu-audit.json",
    "simulated-tpu-campaign.json",
    "simulated-tpu-eager.json",
    "simulated-tpu-jit.json",
    "simulated-tpu-vs-cpu.json",
    "gradient-higher-order.json",
    "gradient-invariants.json",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(arguments: list[str], environment: dict[str, str]) -> None:
    command = [sys.executable, "-m", "xlaxiom.cli", *arguments]
    print(f"[offline-tpu] {' '.join(arguments)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        diagnostics = (completed.stdout + completed.stderr)[-6000:]
        raise RuntimeError(
            f"command exited {completed.returncode}: {' '.join(command)}\n{diagnostics}"
        )


def _all_counts_pass(artifact: dict[str, Any]) -> bool:
    counts = artifact.get("counts", {})
    return bool(counts) and counts.get("pass", 0) == sum(counts.values())


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def collect(output: Path, cases: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    for name in (*ARTIFACT_NAMES, "manifest.json"):
        target = output / name
        if target.exists():
            target.unlink()

    environment = os.environ.copy()
    environment["XLAXIOM_EMULATE_TPU"] = "1"
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )

    eager_path = output / "simulated-tpu-eager.json"
    jit_path = output / "simulated-tpu-jit.json"
    campaign_path = output / "simulated-tpu-campaign.json"
    cpu_path = output / "cpu-jit-evidence.json"
    comparison_path = output / "simulated-tpu-vs-cpu.json"
    higher_order_path = output / "gradient-higher-order.json"
    invariants_path = output / "gradient-invariants.json"

    _run_cli(
        ["property", "run", "--x64", "--backend", "jax-tpu-eager", "--output", str(eager_path)],
        environment,
    )
    _run_cli(
        ["property", "run", "--x64", "--backend", "jax-tpu-jit", "--output", str(jit_path)],
        environment,
    )
    _run_cli(
        [
            "property",
            "campaign",
            "--x64",
            "--cases",
            str(cases),
            "--backend",
            "jax-tpu-jit",
            "--output",
            str(campaign_path),
        ],
        environment,
    )
    _run_cli(
        ["property", "run", "--x64", "--backend", "jax-jit", "--output", str(cpu_path)],
        environment,
    )
    _run_cli(
        [
            "property",
            "compare",
            "--baseline",
            str(cpu_path),
            "--comparison",
            str(jit_path),
            "--output",
            str(comparison_path),
        ],
        environment,
    )
    _run_cli(
        ["gradient", "--higher-order", "--seed", "0", "--output", str(higher_order_path)],
        environment,
    )
    _run_cli(
        ["gradient", "--audit-invariants", "--seed", "0", "--output", str(invariants_path)],
        environment,
    )

    os.environ["XLAXIOM_EMULATE_TPU"] = "1"
    sys.path.insert(0, source_path)
    from xlaxiom.metamorphic import build_property
    from xlaxiom.tpu_harness import audit_tpu_workload_equivalence

    representative = build_property("add-commutativity-v1", seed=0).left
    audit = audit_tpu_workload_equivalence(representative, emulate_if_absent=True)
    audit_path = output / "simulated-tpu-audit.json"
    _write_json(audit_path, audit)

    eager = _load_json(eager_path)
    jit = _load_json(jit_path)
    campaign = _load_json(campaign_path)
    comparison = _load_json(comparison_path)
    higher_order = _load_json(higher_order_path)
    invariants = _load_json(invariants_path)
    checks = {
        "audit_equivalent": audit.get("equivalent") is True,
        "campaign_all_pass": _all_counts_pass(campaign)
        and not campaign.get("candidate_observations"),
        "cpu_comparison_equivalent": comparison.get("summary", {}).get("equivalent") is True,
        "eager_40_of_40": _all_counts_pass(eager) and len(eager.get("properties", [])) == 40,
        "higher_order_pass": higher_order.get("classification") == "pass",
        "invariants_pass": invariants.get("classification") == "pass",
        "jit_40_of_40": _all_counts_pass(jit) and len(jit.get("properties", [])) == 40,
        "simulation_disclosed": audit.get("topology", {}).get("is_simulated") is True,
    }
    artifact_records = {name: _artifact_record(output / name) for name in ARTIFACT_NAMES}
    manifest: dict[str, Any] = {
        "artifacts": artifact_records,
        "cases": cases,
        "checks": checks,
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "offline-tpu-qualification",
        "limitations": [
            "Execution used JAX CPU eager/JIT with explicit TPU simulation metadata.",
            (
                "This does not validate libtpu, TPU compiler lowering, physical collectives, "
                "or TPU performance."
            ),
            "A physical TPU run is required before claiming full-corpus hardware verification.",
        ],
        "mode": "simulated",
        "passed": all(checks.values()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "schema_version": "offline-tpu-qualification-v1",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, default=50, choices=range(1, 101))
    args = parser.parse_args()
    manifest = collect(args.output.resolve(), args.cases)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
