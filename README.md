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

## Built for compiler and platform teams

XLAxiom is useful when a numerical result changes across JAX modes, XLA versions, machines, or accelerators and the team needs an actionable handoff—not a screenshot or an unbounded fuzzer run. It provides a disciplined path from a suspicious result to an engineering-ready artifact:

```text
observe → classify → minimize → reproduce in a fresh process → bundle evidence → compare environments
```

Each handoff can include the minimized declarative case, numerical policy, environment and device details, execution outputs, and checksums. This makes results easier to review in CI, share across teams, attach to an upstream issue, and retest after a compiler or runtime update.

### Responsible escalation

Before filing an upstream JAX/OpenXLA issue, use a real (non-synthetic) case and verify it in a fresh process, record versions and device settings, minimize the predicate, and attach the generated report or validated bundle. A discrepancy is evidence to investigate, never an automatic root-cause or correctness claim.

## When to use it

| Situation | What XLAxiom adds |
| --- | --- |
| A model’s output differs between JAX eager and JIT | A structured comparison, explicit numerical policy, and a minimized predicate. |
| A result changes after a JAX/XLA upgrade | Isolated version-matrix execution and conservative transition classification. |
| A CI or production symptom needs escalation | Portable case, fresh-process verdict, human-readable report, and checksummed bundle. |
| A GPU environment is uncertain or shared | Capability-first validation that reports unsupported or blocked devices instead of silently falling back. |

XLAxiom is **not** a general correctness oracle, performance benchmark suite, or automatic root-cause analyzer. It is the evidence layer between a suspicious numerical result and a focused engineering investigation.

## At a glance

| Need | XLAxiom capability |
| --- | --- |
| Confirm your environment | `xlaxiom doctor` |
| Generate a deterministic workload | `xlaxiom generate` |
| Compare NumPy, JAX eager, and JIT | `xlaxiom compare` |
| Shrink a repeatable mismatch | `xlaxiom minimize` |
| Create a portable evidence record | `xlaxiom report` or `xlaxiom bundle create` |

Read the [artifact guide](docs/artifacts.md) for the recommended handoff package and how to validate or rerun it safely.

## Contents

- [What it does](#what-it-does)
- [Commands](#commands)
- [Cross-version matrix](#cross-version-matrix)
- [Artifact lifecycle](docs/artifacts.md)
- [Architecture](docs/architecture/overview.md)
- [Reproducibility methodology](docs/architecture/reproducibility.md)
- [Backends and precision](#backends-and-precision)
- [Limits and interpretation](#limits-and-interpretation)
- [Development](#development)
- [Roadmap](#roadmap)

## Verified local control

On 2026-10-12, the repository's test environment ran JAX 0.11.2/jaxlib 0.11.2 on Windows CPU (`cpu:0`). A synthetic control starting at 5 operation nodes and 192 input elements was reduced to 1 node and 1 input element in 5 evaluated candidates, then reproduced by a new CLI process. The measured 0.125 output offset is an injected control, not a compiler finding. A portable reduced example is included at [`examples/synthetic-control-reduced.json`](examples/synthetic-control-reduced.json).

### 2026-10-13 EPYC CUDA execution round

The host is an AMD EPYC 7452 (1 socket, 32 cores, 64 threads, 168 GiB, Ubuntu 26.04, kernel 7.0.0-38-generic) with three NVIDIA GPUs under driver 580.178.04 / CUDA 13.0. The primary environment is CPU-only; GPU work runs in the isolated `~/.cache/xlaxiom-gpu/` interpreters with `LD_LIBRARY_PATH` removed per process.

| CUDA ordinal | nvidia-smi | GPU | SM | Variant | Validation | Campaign |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | Tesla V100-SXM2-16GB | 7.0 | `jax[cuda12]` | PASS (4/4) | 8 cases, 16/16 pass |
| 1 | 2 | NVIDIA GeForce RTX 3060 | 8.6 | `jax[cuda13]` | PASS (4/4) | not run — display adapter |
| 2 | 1 | Tesla P40 | 6.1 | `jax[cuda12]` | PASS (4/4) | 8 cases, 16/16 pass |

**Idleness override, recorded.** All three GPUs hold the host's own background workloads (ollama `llama-server`, ComfyUI, VoiceStudio, gnome-shell). The strict idleness gate therefore blocks them by default. Validation was re-run with `--allow-busy`, an explicit operator override that still requires at most 5% utilization and ample free VRAM. The override is justified here because every process on every GPU is the operator's own, utilization was 0-8%, and free VRAM was 7.3-18.6 GiB. The RTX 3060 was not campaigned on: it is the display adapter and its override would have been declined under the rule (8% utilization, four external processes).

**Findings.**
- Bounded GPU campaigns (8 cases, 5 measurements) on the V100 and P40: all 16 NumPy-vs-GPU-JIT and all 16 GPU-eager-vs-GPU-JIT comparisons passed, with placement verified as `cuda:0` on every case. No anomalies.
- JAX #40634 historical study on the V100: `arcsinh(±1e22)` gradients are 0.0 against an analytic reference of 1e-22 (eager and JIT), reproducing known upstream behavior; `acosh(1e22)` passes. Explicitly not a new finding.
- Cross-device consistency (4 checksummed workloads, V100 vs P40): all 4 values identical, placement verified on both devices.
- Metamorphic property corpus on GPU: all 4 properties hold on both GPUs, matching the CPU x64 control within the recorded 2e-06 tolerance.
- CUDA 13 remains unsupported for SM 6.1 and 7.0: the CUDA 13 wheel's `ptxas` has no kernels for those architectures, so `jax[cuda13]` fails with "ptxas too old". This is recorded honestly, not worked around.

**Known issue.** `xlaxiom gpu campaign` must be invoked from inside the isolated interpreter (`<cuda12-probe>/bin/python -m xlaxiom gpu campaign ...`), not via `uv run xlaxiom gpu campaign` with `XLAXIOM_GPU_REEXEC=1` pre-set. Pre-setting the re-exec flag makes `_ensure_gpu_runtime` return early without re-executing into the CUDA environment, so every case reports `unsupported_backend` while the device itself is healthy. Evidence: [`epyc-gpu-campaign-v100-2026-10-13.json`](docs/experiments/epyc-gpu-campaign-v100-2026-10-13.json).

Full machine-readable evidence: [`docs/experiments/`](docs/experiments/).

XLAxiom 0.2 additionally completed a bounded 500-case CPU campaign (seeds 1000–1499) in 39.73 seconds: all 500 NumPy-vs-JIT comparisons and all 500 eager-vs-JIT comparisons passed under the recorded dtype-aware policy. The exact machine-readable record is [available here](docs/experiments/cpu-campaign-2026-10-12.json). It found no new candidate anomalies in that bounded corpus. The [historical replay](docs/research/jax-40634.md) did reproduce known JAX issue #40634 behavior on the locally installed 0.11.2 package; it is explicitly not represented as a new discovery.

Version 0.4 provides a real per-operation historical transition: isolated JAX 0.11.1→0.11.2 testing confirms that `acosh(float32(1e22))` gradient changed from an analytic discrepancy to an oracle pass, while `arcsinh` remains discrepant in both released wheels. See [Historical Regression Study 001](docs/research/historical-regression-study-001.md).

Version 0.5 adds a bounded analytic-gradient campaign. Its local forty-case
CPU record distinguishes two documented large-magnitude `arcsinh`
observations from four unclassified `float32 tanh(10)` candidates; it does not
claim CUDA behavior. See [Gradient Campaign 001](docs/research/gradient-campaign-001.md)
and its [machine-readable record](docs/experiments/gradient-campaign-local.json).

The curated metamorphic corpus includes thirteen reviewed properties, including explicit
float64/x64, mixed-dtype, signed-zero, and non-finite relations. A complete JAX run
must use the `--x64` property option for the float64 relation.

The initial eight-property CPU JIT evidence was recorded on JAX 0.11.2
CPU JIT with seed 42 on 2026-10-13. The checked-in
[evidence record](docs/experiments/metamorphic-corpus-local-2026-10-13.json) includes
both workload identities, execution details, and the numerical policy; it does not
make a broader correctness claim.

The complete thirteen-property x64 CPU JIT run also passed on this host; its
[machine-readable evidence](docs/experiments/metamorphic-corpus-x64-local-2026-10-13.json)
and [coverage record](docs/experiments/metamorphic-property-coverage-x64-2026-10-13.json)
record the exact scope.

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
| `property list` / `property run` | Inspect and execute curated declarative metamorphic properties. |
| `matrix run` | Execute one saved case in isolated pinned JAX environments. |
| `bundle create` / `bundle validate` | Create and validate portable checksummed case bundles. |
| `campaign --isolate` | Run a bounded campaign with each case in a spawned child under a hard timeout. |
| `corpus build` / `validate` | Build and validate the curated reproducible corpus. |
| `corpus metamorphic` | Check every registered metamorphic property against an independent oracle. |

## Cross-version matrix

Matrix definitions are JSON in v0.3 to avoid adding a YAML parser dependency. The checked-in CPU matrix runs JAX 0.11.1 and 0.11.2 in separate `uv`-resolved processes:

```console
uv run xlaxiom matrix run --config configs/matrix-cpu-0.11.json \
  --case examples/synthetic-control-reduced.json --output results/matrix.json
uv run xlaxiom bundle create --case examples/synthetic-control-reduced.json --output control.zip
uv run xlaxiom bundle validate --bundle control.zip
uv run xlaxiom bundle run --bundle control.zip --output results/bundle-evidence.json
```

The matrix report captures environment requests, stdout/stderr, exit status, timeout outcome, and a conservative version classification. On this Windows CPU host, both versions reproduced the explicitly synthetic control; that is a `STABLE_FAILURE`, not a compiler finding. CUDA discovery is capability-gated (`xlaxiom matrix doctor`) and reported unsupported locally.

The bundle workflow validates checksums before execution and runs cases directly from the archive without extraction. That gives reviewers a compact handoff they can validate before running, while keeping the original case identity intact.

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

The evidence foundation is now in place: timeout- and memory-contained workers,
checksummed single and relational artifacts, a reviewed metamorphic corpus, fresh-process
reproduction, CUDA capability validation, and bounded CPU/GPU campaigns. The complete
delivery and verification status—including explicit hardware and external-dependency
boundaries—is tracked in the [holistic delivery plan](docs/roadmap.md).

The corpus deliberately contains no newly discovered compiler finding unless it has a
non-synthetic minimized artifact, fresh-process reproduction, and independently
configured environment evidence. XLAxiom reports unavailable TPU/CUDA hardware as
unsupported; it never converts that absence into a CPU or accelerator success claim.

## Roadmap

Completed:

1. Isolated subprocess workers with per-case timeout enforcement — `xlaxiom campaign --isolate --case-timeout-s N`. A hung kernel or out-of-memory failure now contains one case instead of the whole campaign, and contained failures are recorded rather than silently dropped.
2. CUDA adapter validation on compatible hardware with captured device evidence — `xlaxiom gpu doctor`, `gpu validate`, `gpu campaign`, `gpu historical`. Validated on an AMD EPYC host with a V100 (SM 7.0), an RTX 3060 (SM 8.6), and a P40 (SM 6.1). See [the multi-GPU campaign evidence](docs/experiments/epyc-multi-gpu-campaign.md). TPU remains unvalidated because no TPU hardware is available here; it is reported unsupported rather than assumed.
3. A curated corpus of independently reproduced findings and metamorphic properties — `xlaxiom corpus build`, `corpus validate`, `corpus metamorphic`. A finding is admitted only after it reproduces in at least two fresh processes, and the admission bar rejects an entry whose reproduction evidence is missing or whose checksum no longer matches.
4. Metamorphic properties on CUDA. The property suite now runs on explicitly selected CUDA GPUs. On 2026-10-13 all four properties held identically on a Tesla V100 (SM 7.0, `jax[cuda12]`) and a Tesla P40 (SM 6.1, `jax[cuda12]`), and matched a CPU x64 control within the recorded 2e-06 tolerance. Evidence: [`epyc-metamorphic-crossbackend-2026-10-13.json`](docs/experiments/epyc-metamorphic-crossbackend-2026-10-13.json). The RTX 3060 (SM 8.6, `jax[cuda13]`) is excluded from the comparison because it is the host display adapter with an external workload and a materially different configuration; comparing it would be inconclusive by design.

Next:

1. **Minimize admitted corpus findings automatically.** `corpus build` admits a reproduced finding with its reduced workload, but does not yet run the failure-preserving minimizer over the corpus end to end. Wiring `minimize` into the corpus pipeline would shrink admitted cases to their smallest reproducing form.
2. **Cross-backend corpus comparison.** `gradient-compare` already compares two campaigns. Extending the same treatment to the corpus would let a CPU baseline and a CUDA run be diffed entry by entry, separating shared candidates from accelerator-specific ones automatically.
3. **More metamorphic relations.** The suite holds four properties. Precision-sensitive relations, such as a reduction commuting with a dtype change, or `sin`/`cos` identities holding under argument reduction, would widen coverage.
4. **Longer bounded corpus campaigns.** A wider seed sweep would raise the chance of admitting a genuinely new finding rather than confirming that known behavior still reproduces.
