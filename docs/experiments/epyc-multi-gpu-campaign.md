# XLAxiom EPYC multi-GPU campaign — evidence and environment notes

## Environment (exact)

| Item | Value |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS |
| Kernel | Linux 7.0.0-38-generic x86_64 |
| CPU | AMD EPYC 7452 32-core, 64 threads |
| RAM | 168 GiB |
| NVIDIA driver | 580.178.04 |
| CUDA runtime | 13.0 |
| System CUDA toolkit | 12.6 |
| Python (primary env) | 3.12.13 |
| Python (isolated GPU envs) | 3.12.13 |
| JAX / jaxlib | 0.11.2 / 0.11.2 |

## Device identity — read this before selecting a GPU

**CUDA runtime ordinal order does not match `nvidia-smi` index order on this host.**

| CUDA ordinal | `nvidia-smi` index | GPU | SM | Correct JAX wheel |
| --- | --- | --- | --- | --- |
| 0 | 0 | Tesla V100-SXM2-16GB | 7.0 | `jax[cuda12]` |
| 1 | 2 | NVIDIA GeForce RTX 3060 | 8.6 | `jax[cuda13]` |
| 2 | 1 | Tesla P40 | 6.1 | `jax[cuda12]` |

`CUDA_VISIBLE_DEVICES` uses the **ordinal**, not the `nvidia-smi` index. An operator who
runs `nvidia-smi`, sees the P40 at index 1, and sets `CUDA_VISIBLE_DEVICES=1` silently
targets the RTX 3060. `xlaxiom gpu doctor` now reports this mismatch explicitly.

### Second trap: `LD_LIBRARY_PATH` shadowing

The host exports `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64`. That takes precedence over
the CUDA libraries bundled inside the JAX wheels and shadows the newer cuSPARSE with the
older system copy, so plugin initialization fails with `Unable to load cuSPARSE` naming a
library the wheel actually ships.

Fix, per-process only — **do not** change the host driver, the system CUDA installation,
or any running service:

```console
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=<ordinal> <interpreter> -m xlaxiom ...
```

`xlaxiom gpu doctor` reports this risk too.

## Campaign results (24 cases, 5 measurements each)

| CUDA ordinal | GPU | SM | Variant | Cases | Passing checks | Anomalies |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Tesla V100-SXM2-16GB | 7.0 | `jax[cuda12]` | 24 | 48 | 0 |
| 2 | Tesla P40 | 6.1 | `jax[cuda12]` | 24 | 48 | 0 |

Both devices pass all 24 NumPy-vs-GPU-JIT and all 24 GPU-eager-vs-GPU-JIT comparisons on
`cuda:0`, with CPU and GPU timings recorded separately and XLA compilation timed apart
from steady-state execution.

Measured on the V100 under `jax[cuda12]`:

| Family | CPU median | GPU median | GPU compile |
| --- | --- | --- | --- |
| matmul | ~0.014 ms | ~0.151 ms | ~35.8 ms |
| broadcasting | ~0.014 ms | ~0.162 ms | ~37.0 ms |
| reduction | ~0.012 ms | ~0.138 ms | ~44.4 ms |
| precision_sensitive | ~0.013 ms | ~0.156 ms | ~36.0 ms |

CPU is faster on these deliberately tiny bounded graphs. That is expected at this scale
and is reported as measured, not adjusted.

### RTX 3060 (ordinal 1) — BLOCKED, not failed

The RTX 3060 is the host's display GPU and was at 6% utilization with an external Python
workload resident. The strict idleness gate therefore blocks it: `--allow-busy` is an
explicit operator override and still requires at most 5% utilization and ample free VRAM.

This is the safety rule working as designed. No probe was run against a device with an
active external workload. Earlier probing of this GPU passed all four bounded checks under
`jax[cuda13]` when it was quiet; `ptxas` in the CUDA 13 wheel has no SM 6.1 or 7.0 kernels,
which is why CUDA 13 is correctly refused for the V100 and P40.

## Subprocess isolation

`xlaxiom campaign --isolate --case-timeout-s N` now runs every case in a spawned child
under a hard timeout, so a hung kernel or an out-of-memory failure contains one case
instead of taking down the whole campaign. Contained failures are recorded in
`contained_failures` and counted in `isolation.contained_count`; they are never silently
retried or reported as passes.

Verified: a worker sleeping 5 s under a 0.5 s timeout is terminated and reported as
`timeout`, and the campaign continues.

## Reproduction

```console
uv sync --extra dev

# capability and environment risk report (no GPU execution)
uv run xlaxiom gpu doctor

# per-device bounded probes
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  <cuda12-interpreter> -m xlaxiom gpu validate --allow-busy

# 24-case campaign on the V100
env -u LD_LIBRARY_PATH XLAXIOM_GPU_REEXEC=1 CUDA_VISIBLE_DEVICES=0 \
  <cuda12-interpreter> -m xlaxiom gpu campaign --allow-busy \
  --cases 24 --measurements 5 --output results/gpu-campaign-ord0.json

# isolated campaign with per-case timeout
uv run xlaxiom campaign --cases 20 --isolate --case-timeout-s 30
```

Invoke via `-m xlaxiom`, never `-m xlaxiom.cli`.

No secrets, absolute personal paths, SSH keys, or raw GPU UUIDs are recorded. GPU UUIDs
are redacted to a salted digest.
