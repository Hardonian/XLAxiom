# XLAxiom — turn numerical discrepancies into evidence

[![CI](https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/)

**XLAxiom turns a suspicious JAX/OpenXLA numerical result into a minimized, reproducible, portable, evidence-backed case.** It gives compiler, platform, and ML infrastructure teams a disciplined path from “this changed” to an artifact another engineer can inspect, rerun, and compare.

It is independently developed and is not affiliated with, sponsored by, or endorsed by Google.

> **Portfolio signal:** XLAxiom is the ML-infrastructure flagship: a reproducible testing and evidence pipeline for turning JAX/OpenXLA numerical discrepancies into small, reviewable engineering cases.

## Pick your path

| If you need to… | Start here |
| --- | --- |
| Prove the pipeline works safely | [Run the five-minute synthetic control](#five-minute-confidence-check). |
| Investigate a real numerical discrepancy | Follow the [artifact handoff workflow](docs/artifacts.md). |
| Add bounded regression coverage to CI | Run the [curated property corpus](docs/corpus.md). |
| Compare CPU, GPU, or version-specific evidence | Use the [support matrix](docs/support.md) and [quickstart](docs/quickstart.md). |
| Inspect physical TPU qualification | Review the [retained 4-chip TPU evidence](evidence/tpu/manifest.json). |
| Qualify TPU workflows without accelerator cost | Run the [offline TPU qualification](docs/offline-tpu.md). |
| Integrate with Google Cloud, TPUs & Colab | Read the [Google Ecosystem guide](docs/google-ecosystem.md). |
| Install from source or a verified wheel | Read the [installation guide](docs/installation.md). |

## Five-minute confidence check

Prerequisite: Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). From a source checkout:

```console
uv sync --extra dev --frozen
uv run xlaxiom doctor
uv run xlaxiom generate --synthetic --seed 7 --output results/original.json
uv run xlaxiom minimize --case results/original.json --output results/reduced.json
uv run xlaxiom reproduce --case results/reduced.json
uv run xlaxiom report --case results/reduced.json --output results/report
uv run xlaxiom bundle create --case results/reduced.json --output results/control.zip
uv run xlaxiom bundle validate --bundle results/control.zip
```

This produces a small, checksummed control case, a fresh-process reproduction verdict, a JSON/Markdown report, and a validated portable bundle. `--synthetic` injects a clearly labelled offset after JAX execution: it proves the workflow, but is never evidence of a JAX, XLA, hardware, or model defect.

For a real observation, start with the [artifact guide](docs/artifacts.md). It explains the required minimization, fresh-process reproduction, environment evidence, and safe handoff boundaries before an upstream issue is considered.

## What XLAxiom does

```text
observe → classify → minimize → reproduce in a fresh process
        → bundle evidence → compare environments
```

- **Records the evidence.** It captures versions, device placement, numerical policy, outputs, and an explicit predicate instead of treating a mismatch as a diagnosis.
- **Keeps cases small.** Failure-preserving minimization and topological subgraph extraction reduce a reproducible observation without silently changing what failed.
- **Pinpoints compiler regressions.** Automated bisection binary-searches compiler releases in isolated environments to identify exact breaking changes.
- **Makes handoffs portable.** Declarative JSON workloads, copy-pasteable 10-line reproduction scripts, triage markdown packages, checksummed reports, and bundles let another engineer or upstream maintainer validate the observation immediately.
- **Fails closed on hardware.** Accelerator requests must prove actual placement; unavailable GPU/TPU hardware is never represented as a CPU success.

XLAxiom is the evidence layer between a suspicious numerical result and a focused engineering investigation. It is not a general correctness oracle, performance benchmark suite, or automatic root-cause analyzer.

## The normal workflow

1. **Capture an observation.** Compare NumPy against JAX eager/JIT or execute a reviewed relation with an explicit policy.
2. **Reduce it without changing it.** Minimize a repeatable non-synthetic predicate and preserve its reduction record.
3. **Reproduce it independently.** Re-run in a fresh process, then record a second host, device, or runtime when the claim requires it.
4. **Hand it off safely.** Generate a report and validated bundle; attach only the evidence that supports the stated scope.

The [artifact lifecycle](docs/artifacts.md) has the exact command-level process. The [issue readiness check](docs/quickstart.md#before-escalating-a-real-observation) is deliberately conservative: an observation is not a diagnosis, and missing external evidence is never inferred.

## Evidence you can trust

The repository is designed to make scope visible rather than inflate it:

- The CPU path, checksummed artifacts, failure containment, and reviewed property corpus are covered by automated tests and release verification.
- The curated corpus contains forty explicit relations across ordinary, extreme, signed-zero, non-finite, integer, mixed-dtype, and x64 inputs with verified mutation sharpness.
- Windows/Linux, selected CUDA, and 4-chip Google Cloud TPU v5e evidence are retained as concrete records; untested hardware remains capability- and evidence-gated.

See the [support and verification matrix](docs/support.md) for what is supported versus merely implemented, and the [experiment records](docs/experiments/) for the underlying evidence—not a condensed lab notebook in this README.

## Command map

| Goal | Primary commands |
| --- | --- |
| Ingest live code | `trace` |
| Inspect an environment | `doctor`, `environment capture`, `runtime compare` |
| Investigate one case | `generate`, `run`, `compare`, `minimize`, `reproduce` |
| Pinpoint compiler regression | `bisect` |
| Prepare upstream triage/handoff | `triage`, `report`, `bundle create`, `bundle validate`, `readiness` |
| Test known relations | `property list`, `property run`, `property campaign`, `relation` |
| Compare changes safely | `matrix run`, `property compare`, `historical-study` |
| Validate accelerators | `gpu doctor`, `gpu validate`, `gpu campaign` |
| Google Ecosystem & Cloud TPUs | `cloud tpu-run`, `cloud tpu-status`, `cloud vertex-spec`, `cloud xpk-spec` |
| Developer surfaces & Colab | `colab`, `xla pass-isolate` |
| Enterprise Governance & Telemetry | `bigquery schema`, `bigquery export`, `kms sign`, `kms verify` |

Run `xlaxiom <command> --help` for the complete interface. The [documentation index](docs/index.md) routes each workflow to the right level of detail.

## Safety and interpretation

Workloads are deterministic JSON data selected from a closed operation registry; XLAxiom does not load pickle files or execute workload-supplied code. Numerical workloads can still consume meaningful resources, so run untrusted artifacts with suitable process and resource isolation. Read [SECURITY.md](SECURITY.md) before sharing artifacts or reporting a vulnerability.

NumPy is useful but is not an independent oracle for every JAX semantic, dtype-promotion rule, compiler setting, or hardware implementation. A mismatch directs investigation; it never establishes root cause or severity by itself.

## Development and release

```console
uv sync --extra dev --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build --out-dir .release-dist
uv run twine check .release-dist/*
uv run python scripts/verify_distribution.py --dist-dir .release-dist
```

The [release policy](docs/release.md) defines the required quality gates, reproducible-build expectations, and Trusted Publishing boundary. The [roadmap](docs/roadmap.md) distinguishes implemented capability from physical-hardware evidence that cannot be manufactured locally.
