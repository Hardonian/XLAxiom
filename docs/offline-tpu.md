# Offline TPU qualification

Use the offline qualification workflow when physical TPU access is unavailable or not
cost-effective:

```console
uv run python scripts/collect_tpu_offline_evidence.py
```

The workflow runs the complete 40-property corpus in eager and JIT modes with JAX x64,
a 50-seed simulated TPU campaign, CPU-versus-simulated-TPU classification comparison,
higher-order gradient checks, and a representative topology audit. It writes checksummed
artifacts plus `evidence/tpu-offline/manifest.json`.

The manifest deliberately records `mode: simulated`, and every simulated execution retains
`simulation_backend: cpu`. This evidence validates XLAxiom's workload construction, x64
handling, comparison logic, campaign orchestration, and evidence pipeline at zero accelerator
cost. It does **not** validate libtpu, physical TPU compiler lowering, inter-chip collectives,
or accelerator performance. Those claims remain gated on a bounded physical run.

Use fewer campaign cases for a quick development check:

```console
uv run python scripts/collect_tpu_offline_evidence.py --cases 5 --output .verification-results/tpu-offline
```

Only evidence under `evidence/tpu-offline/` is intended for retention. Temporary development
runs should remain under `.verification-results/`.
