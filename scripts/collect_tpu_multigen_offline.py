"""Generate multi-generation TPU qualification evidence suites.

Produces offline qualification packages for TPU v4, TPU v5p, and TPU v6e Trillium
with explicit simulation disclosure, hardware topology manifests, and SHA-256 checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xlaxiom.cloud.tpu import TPU_GENERATIONS

ROOT = Path(__file__).resolve().parents[1]

GENERATIONS = {
    "v4": {
        "output": ROOT / "evidence" / "tpu-v4",
        "accelerator": "v4-8",
        "topology": "2x2x1 3D Torus",
        "chips": 4,
        "memory_per_chip_gb": 32,
    },
    "v5p": {
        "output": ROOT / "evidence" / "tpu-v5p",
        "accelerator": "v5p-8",
        "topology": "2x2x2 3D Torus",
        "chips": 4,
        "memory_per_chip_gb": 95,
    },
    "v6e": {
        "output": ROOT / "evidence" / "tpu-v6e",
        "accelerator": "v6e-4",
        "topology": "2x2 2D Mesh",
        "chips": 4,
        "memory_per_chip_gb": 32,
    },
}

ARTIFACT_NAMES = (
    "cpu-jit-evidence.json",
    "simulated-tpu-audit.json",
    "simulated-tpu-campaign.json",
    "simulated-tpu-eager.json",
    "simulated-tpu-jit.json",
    "simulated-tpu-vs-cpu.json",
    "gradient-higher-order.json",
    "gradient-invariants.json",
    "hardware-topology.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cli(arguments: list[str], environment: dict[str, str]) -> None:
    command = [sys.executable, "-m", "xlaxiom.cli", *arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        diagnostics = (completed.stdout + completed.stderr)[-3000:]
        raise RuntimeError(
            f"command exited {completed.returncode}: {' '.join(command)}\n{diagnostics}"
        )


def collect_generation(gen_key: str, cases: int = 5) -> dict[str, Any]:
    cfg = GENERATIONS[gen_key]
    output: Path = cfg["output"]
    output.mkdir(parents=True, exist_ok=True)

    for name in (*ARTIFACT_NAMES, "manifest.json"):
        p = output / name
        if p.exists():
            p.unlink()

    environment = os.environ.copy()
    environment["XLAXIOM_EMULATE_TPU"] = "1"
    environment["XLAXIOM_TPU_GENERATION"] = gen_key
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
    topo_path = output / "hardware-topology.json"

    # Write generation topology spec
    from dataclasses import asdict

    raw_spec = TPU_GENERATIONS.get(cfg["accelerator"].split("-")[0])
    gen_spec = asdict(raw_spec) if raw_spec is not None else {}
    topo_data = {
        "generation": gen_key,
        "accelerator": cfg["accelerator"],
        "topology": cfg["topology"],
        "chips": cfg["chips"],
        "memory_per_chip_gb": cfg["memory_per_chip_gb"],
        "architecture_details": gen_spec,
    }
    _write_json(topo_path, topo_data)

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

    sys.path.insert(0, source_path)
    from xlaxiom.metamorphic import build_property
    from xlaxiom.tpu_harness import audit_tpu_workload_equivalence

    representative = build_property("add-commutativity-v1", seed=0).left
    audit = audit_tpu_workload_equivalence(representative, emulate_if_absent=True)
    audit_path = output / "simulated-tpu-audit.json"
    _write_json(audit_path, audit)

    # Build manifest
    artifacts = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha256(output / name)}
        for name in ARTIFACT_NAMES
    }
    manifest = {
        "schema_version": "offline-tpu-qualification-v1",
        "kind": f"offline-tpu-{gen_key}-qualification",
        "generation": gen_key,
        "accelerator": cfg["accelerator"],
        "topology": cfg["topology"],
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "simulated",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "passed": True,
        "limitations": [
            f"Simulated TPU {gen_key} ({cfg['accelerator']}) on CPU with simulation metadata.",
            "Validates DAG generation, shape invariants, tolerances, and collective mesh bindings.",
            "Physical TPU allocation is required before claiming full hardware verification.",
        ],
        "artifacts": artifacts,
    }
    _write_json(output / "manifest.json", manifest)
    print(f"Generated qualification suite for TPU {gen_key} in {output}", flush=True)
    return manifest


def main() -> int:
    for gen in ("v4", "v5p", "v6e"):
        print(f"Generating TPU {gen} offline qualification suite...", flush=True)
        collect_generation(gen, cases=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
