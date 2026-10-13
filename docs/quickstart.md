# First five minutes with XLAxiom

XLAxiom has two distinct entry points: a safe synthetic control that proves your local
pipeline, and a real-observation workflow that produces reviewable evidence. Do not
mistake the former for a compiler or hardware finding.

## Confirm the local CPU path

From a source checkout with Python 3.11–3.13 and `uv` installed:

```console
uv sync --extra dev --frozen
uv run xlaxiom doctor
uv run xlaxiom generate --synthetic --seed 7 --output results/original.json
uv run xlaxiom minimize --case results/original.json --output results/reduced.json
uv run xlaxiom reproduce --case results/reduced.json
uv run xlaxiom report --case results/reduced.json --output results/control-report
uv run xlaxiom bundle create --case results/reduced.json --output results/control.zip
uv run xlaxiom bundle validate --bundle results/control.zip
```

The expected classification is a visibly labelled synthetic discrepancy. The command
sequence demonstrates deterministic generation, failure-preserving minimization,
fresh-process reproduction, reporting, and safe bundle validation; it is not evidence
of a JAX, XLA, hardware, or model defect.

## Run a bounded regression check

The curated corpus exercises reviewed relations and records the exact property seed,
workload identities, backend, and numerical policy:

```console
uv run xlaxiom property coverage
uv run xlaxiom property run --property matmul-transpose-v1 --seed 42 \
  --backend jax-jit --x64 --output results/cpu-property.json
uv run xlaxiom property campaign --cases 20 --backend jax-jit --x64 \
  --output results/cpu-campaign.json
```

Use the single-property evidence file when you want to compare the identical generated
workloads elsewhere. Use the campaign record to monitor the bounded local corpus; it is
not an input to `property compare`.

## Compare two real evidence artifacts

`property compare` only accepts two artifacts that record the same property selection,
seed, ordering, and workload hashes. Create them first—one per execution environment.

On the CPU host:

```console
uv run xlaxiom property run --property matmul-transpose-v1 --seed 42 \
  --backend jax-jit --x64 --output results/cpu-property.json
```

On a separately validated GPU host:

```console
xlaxiom gpu doctor
xlaxiom gpu validate --output results/gpu-validation.json
xlaxiom property run --property matmul-transpose-v1 --seed 42 \
  --backend jax-gpu-jit --x64 --output results/gpu-property.json
```

Bring the two evidence files together, then compare them:

```console
uv run xlaxiom property compare --baseline results/cpu-property.json \
  --comparison results/gpu-property.json --output results/cpu-gpu-comparison.json
```

An unavailable or incompatible accelerator must remain `unsupported_backend`; it is
never a CPU fallback. See the [support matrix](support.md) for the difference between
supported, verified, and hardware-unverified paths.

## Before escalating a real observation

Use a non-synthetic, minimized case and preserve the exact command output from a
fresh-process reproduction. Record each independent host or device with
`environment capture`, generate a report, validate the bundle, and then run
`xlaxiom readiness` against the case and supplied evidence. The readiness result is a
conservative gate, not an automatic issue filing tool.

The [artifact guide](artifacts.md) explains the full handoff package and the
[security policy](../SECURITY.md) covers private-machine details and vulnerability
reports.

