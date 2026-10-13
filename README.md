# XLAxiom — evidence-first testing for JAX and OpenXLA

[![CI](https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/)

**Find, reduce, and reproduce numerical and compiler discrepancies in JAX/OpenXLA workloads.** XLAxiom is an open-source differential-testing laboratory that turns an observation into a small, portable, evidence-backed case.

It combines deterministic declarative workload generation, NumPy/JAX and eager/JIT comparison, failure-preserving minimization, and checksummed reports—so a potential issue is easier to validate, share, and investigate.

It is independently developed and is not affiliated with, sponsored by, or endorsed by Google.

## Start here

Install the development environment, confirm your JAX setup, then create and minimize a safe synthetic control:

```console
uv sync --extra dev
uv run xlaxiom doctor
uv run xlaxiom generate --synthetic --seed 7 --output results/original.json
uv run xlaxiom minimize --case results/original.json --output results/reduced.json
uv run xlaxiom reproduce --case results/reduced.json
uv run xlaxiom report --case results/reduced.json --output results/report
```

`--synthetic` injects a clearly labelled output offset after JAX execution. It proves the testing pipeline; it is never evidence of a JAX, XLA, or hardware defect. Omit it for genuine NumPy-versus-JAX or JAX-eager-versus-JIT observations.

## Why XLAxiom?

- **Evidence first:** records versions, devices, policies, outputs, and a reproducible predicate instead of treating a mismatch as a diagnosis.
- **Smallest useful case:** reduces a reproducible observation while preserving the failure signature.
- **Portable by design:** workloads are deterministic JSON data, with checksummed bundles and fresh-process reproduction.
- **Conservative claims:** distinguishes synthetic controls, observations, historical behavior, and independently reproduced findings.

## At a glance

| Need | XLAxiom capability |
| --- | --- |
| Confirm your environment | `xlaxiom doctor` |
| Generate a deterministic workload | `xlaxiom generate` |
| Compare NumPy, JAX eager, and JIT | `xlaxiom compare` |
| Shrink a repeatable mismatch | `xlaxiom minimize` |
| Create a portable evidence record | `xlaxiom report` or `xlaxiom bundle create` |

## Contents

- [What it does](#what-it-does)
- [Commands](#commands)
- [Cross-version matrix](#cross-version-matrix)
- [Backends and precision](#backends-and-precision)
- [Limits and interpretation](#limits-and-interpretation)
- [Development](#development)
- [Roadmap](#roadmap)

## Verified local control

On 2026-10-12, the repository's test environment ran JAX 0.11.2/jaxlib 0.11.2 on Windows CPU (`cpu:0`). A synthetic control starting at 5 operation nodes and 192 input elements was reduced to 1 node and 1 input element in 5 evaluated candidates, then reproduced by a new CLI process. The measured 0.125 output offset is an injected control, not a compiler finding. A portable reduced example is included at [`examples/synthetic-control-reduced.json`](examples/synthetic-control-reduced.json).

XLAxiom 0.2 additionally completed a bounded 500-case CPU campaign (seeds 1000–1499) in 39.73 seconds: all 500 NumPy-vs-JIT comparisons and all 500 eager-vs-JIT comparisons passed under the recorded dtype-aware policy. The exact machine-readable record is [available here](docs/experiments/cpu-campaign-2026-10-12.json). It found no new candidate anomalies in that bounded corpus. The [historical replay](docs/research/jax-40634.md) did reproduce known JAX issue #40634 behavior on the locally installed 0.11.2 package; it is explicitly not represented as a new discovery.

Version 0.4 provides a real per-operation historical transition: isolated JAX 0.11.1→0.11.2 testing confirms that `acosh(float32(1e22))` gradient changed from an analytic discrepancy to an oracle pass, while `arcsinh` remains discrepant in both released wheels. See [Historical Regression Study 001](docs/research/historical-regression-study-001.md).

Version 0.5 adds a bounded analytic-gradient campaign. Its local forty-case
CPU record distinguishes two documented large-magnitude `arcsinh`
observations from four unclassified `float32 tanh(10)` candidates; it does not
claim CUDA behavior. See [Gradient Campaign 001](docs/research/gradient-campaign-001.md)
and its [machine-readable record](docs/experiments/gradient-campaign-local.json).

## What it does

```text
declarative workload -> NumPy/JAX execution -> explicit numerical policy
    -> classified observation -> failure-preserving minimization
    -> checksummed JSON case -> fresh-process reproduction -> JSON/Markdown report
```

The initial safe operation registry includes elementwise arithmetic, `matmul`, reshape, transpose, reductions, and selected unary functions. Workloads are JSON data, not executable Python, and their input data is reconstructed deterministically from a recorded seed.

## Commands

| Command | Purpose |
| --- | --- |
| `doctor` | Report JAX installation, devices, and a real CPU probe. |
| `generate` | Create a deterministic portable workload case. |
| `run` | Execute NumPy, JAX eager, or JAX JIT. |
| `compare` | Compare NumPy/JAX or eager/JIT outputs. |
| `minimize` | Reduce a currently reproducible observation. |
| `reproduce` | Reexecute and verify the saved predicate. |
| `report` | Produce evidence JSON and readable Markdown. |
| `benchmark` | Run a bounded warmup plus timing sample. |
| `historical` | Replay trusted adapters for documented historical regressions. |
| `gradient` / `gradient-campaign` | Run one or a bounded set of analytic-gradient checks. |
| `gradient-compare` | Compare two self-identifying gradient campaign artifacts. |
| `campaign` | Run a duration-bounded trusted CPU discovery campaign. |
| `matrix run` | Execute one saved case in isolated pinned JAX environments. |
| `bundle create` / `bundle validate` | Create and validate portable checksummed case bundles. |

## Cross-version matrix

Matrix definitions are JSON in v0.3 to avoid adding a YAML parser dependency. The checked-in CPU matrix runs JAX 0.11.1 and 0.11.2 in separate `uv`-resolved processes:

```console
uv run xlaxiom matrix run --config configs/matrix-cpu-0.11.json \
  --case examples/synthetic-control-reduced.json --output results/matrix.json
uv run xlaxiom bundle create --case examples/synthetic-control-reduced.json --output control.zip
uv run xlaxiom bundle validate --bundle control.zip
```

The matrix report captures environment requests, stdout/stderr, exit status, timeout outcome, and a conservative version classification. On this Windows CPU host, both versions reproduced the explicitly synthetic control; that is a `STABLE_FAILURE`, not a compiler finding. CUDA discovery is capability-gated (`xlaxiom matrix doctor`) and reported unsupported locally.

All commands expose `--help`. `reproduce` runs the workload again; it does not merely display saved output.

## Backends and precision

NumPy and JAX CPU are the supported initial backends. JAX’s CPU wheel is supported on Windows, but native Windows GPU support is not; accelerator support is a runtime capability question on a target host, not a claim made by this project. The active execution device is included in observations.

JAX defaults to 32-bit types unless its x64 option is enabled. XLAxiom therefore records dtype and applies the case policy explicitly. Its default policy uses both relative and absolute tolerances, compares NaN location and infinity sign, and detects signed-zero differences. It does not use a hidden universal tolerance or silently relax a failing case.

## Limits and interpretation

The generator limits graph size, input count, and tensor elements. The minimizer has a bounded candidate budget and reexecutes every accepted transformation. Current in-process execution is appropriate for bounded local workloads; future releases should use subprocess timeouts for untrusted or resource-intensive campaigns.

NumPy is useful but is not an independent correctness oracle for every JAX semantic, dtype-promotion rule, compiler setting, or hardware implementation. A discrepancy is an observation, not a root-cause claim. Before filing upstream, reproduce it in a fresh process and independently configured environment, eliminate synthetic controls, record versions/devices/settings, and minimize it.

## Development

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
```

See [the architecture](docs/architecture/overview.md), [reproducibility methodology](docs/architecture/reproducibility.md), [research landscape](docs/research/landscape.md), and [contributing guide](CONTRIBUTING.md).

## Roadmap

1. Add isolated subprocess workers and timeout/memory enforcement.
2. Validate CUDA/TPU adapters only on compatible hardware and capture device evidence.
3. Build a curated corpus of independently reproduced, minimized findings and metamorphic properties.
