# XLAxiom

Finding the smallest proof of a compiler failure.

XLAxiom is an open-source laboratory for discovering, minimizing, and reproducing numerical and compiler discrepancies in JAX/OpenXLA workloads. It pairs deterministic declarative workload generation with NumPy/JAX comparison, failure-preserving reduction, and portable evidence reports.

It is independently developed and is not affiliated with, sponsored by, or endorsed by Google.

## Quick start

```console
uv sync --extra dev
uv run xlaxiom doctor
uv run xlaxiom generate --synthetic --seed 7 --output results/original.json
uv run xlaxiom minimize --case results/original.json --output results/reduced.json
uv run xlaxiom reproduce --case results/reduced.json
uv run xlaxiom report --case results/reduced.json --output results/report
```

`--synthetic` injects a clearly labelled output offset after JAX execution. It proves the testing pipeline; it is never evidence of a JAX, XLA, or hardware defect. Omit it for genuine NumPy-versus-JAX or JAX-eager-versus-JIT observations.

## Verified local control

On 2026-10-12, the repository's test environment ran JAX 0.11.2/jaxlib 0.11.2 on Windows CPU (`cpu:0`). A synthetic control starting at 5 operation nodes and 192 input elements was reduced to 1 node and 1 input element in 5 evaluated candidates, then reproduced by a new CLI process. The measured 0.125 output offset is an injected control, not a compiler finding. A portable reduced example is included at [`examples/synthetic-control-reduced.json`](examples/synthetic-control-reduced.json).

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
