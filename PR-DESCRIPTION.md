# XLAxiom EPYC CUDA hardware validation — evidence summary

This PR adds real, bounded CUDA validation for XLAxiom on an AMD EPYC host with three
NVIDIA GPUs, and records portable evidence artifacts. It adds no CPU fallback path: a
requested accelerator that is absent, busy, or numerically unsupported is reported as
UNSUPPORTED or BLOCKED, never replaced by a CPU result.

## Environment (exact)

| Item | Value |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS |
| Kernel | Linux 7.0.0-38-generic x86_64 |
| CPU | AMD EPYC 7452 32-core, 64 threads |
| RAM | 168 GiB |
| NVIDIA driver | 580.178.04 |
| CUDA runtime (driver) | 13.0 |
| System CUDA toolkit | 12.6 (`/usr/local/cuda-12.6`) |
| Python (primary env) | 3.12.13 |
| Python (isolated GPU envs) | 3.12.13 |
| JAX | 0.11.2 |
| jaxlib | 0.11.2 |
| JAX CUDA wheels | `jax[cuda12]==0.11.2`, `jax[cuda13]==0.11.2` |
| Hostname | epyc |

## GPUs and verified compatibility

Compatibility follows the official JAX installation rules: CUDA 12 wheels require SM 5.2
or newer with a Linux driver >= 525; CUDA 13 wheels require SM 7.5 or newer with a Linux
driver >= 580.

| CUDA ordinal | nvidia-smi index | GPU | SM | Variant used | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | Tesla V100-SXM2-16GB | 7.0 | `jax[cuda12]` | PASS |
| 1 | 2 | NVIDIA GeForce RTX 3060 | 8.6 | `jax[cuda13]` | PASS |
| 2 | 1 | Tesla P40 | 6.1 | `jax[cuda12]` | PASS |

Empirically probed CUDA 13 behaviour (see `docs/experiments/epyc-cuda13-compatibility.json`):

- V100 (SM 7.0) — FAIL: `ptxas` does not support CC 7.0, so CUDA 13 is correctly avoided.
- P40 (SM 6.1) — FAIL: `ptxas` does not support CC 6.1, so CUDA 12 is correctly used.
- RTX 3060 (SM 8.6) — PASS under CUDA 13.

### Two environment root causes found

1. **`LD_LIBRARY_PATH` shadowing.** The host exports `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64`,
   which shadows the newer cuSPARSE bundled inside the JAX CUDA 12 wheel with the older
   system copy. `jax[cuda12]` then fails with `Unable to load cuSPARSE`. The fix is
   per-process (`env -u LD_LIBRARY_PATH`) and never modifies the host driver, the system
   CUDA installation, or any service.

2. **CUDA ordinal order differs from `nvidia-smi` order.** The CUDA runtime enumerates
   `0=V100, 1=RTX 3060, 2=P40` while `nvidia-smi` reports `0=V100, 1=P40, 2=RTX 3060`.
   Selecting "device 1" therefore targets the RTX 3060, not the P40. Device identity is
   now resolved by matching the CUDA runtime's reported name, and every artifact records
   both the CUDA ordinal and the `nvidia-smi` index.

## Capability evidence

`xlaxiom doctor` (isolated CUDA 12 environment, V100):

```json
{
  "cpu_probe": "ok",
  "devices": ["cuda:0"],
  "gpu_devices": ["cuda:0"],
  "gpu_probe": "ok",
  "jax": "0.11.2",
  "jaxlib": "0.11.2",
  "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.43",
  "python": "3.12.13"
}
```

`xlaxiom matrix doctor`:

```json
{
  "cuda": {
    "devices": ["cuda:0"],
    "jaxlib": "0.11.2",
    "status": "available",
    "supported": true
  },
  "format": "JSON matrix config only in v0.3"
}
```

## Gradient campaign (CUDA evidence)

Produced with the repository's own command on the V100 under `jax[cuda12]`:

```
uv run xlaxiom gradient-campaign --cases 40 --seed 0 \
  --output results/gradient-campaign-epyc-cuda.json
```

- `environment.devices`: `["cuda:0"]` — this is real CUDA evidence, not a CPU run.
- JAX/jaxlib: 0.11.2 / 0.11.2, Python 3.12.13
- Counts: 34 pass, 2 known historical discrepancy, 4 candidate discrepancy
- Operation coverage: acosh 10, arcsinh 10, exp 10, tanh 10
- `candidate_anomalies`: 4

The two `arcsinh`/`acosh` discrepancies at extreme magnitudes are documented upstream
behaviour from JAX issue #40634 and are segregated as known, not reported as new findings.

## Verification

| Gate | Result |
| --- | --- |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 50 files already formatted |
| `uv run mypy src` | Success: no issues found in 24 source files |
| `uv run pytest -q` | 39 passed, 1 skipped |
| `uv build` | wheel + sdist built successfully |

The single skip is `tests/unit/test_gpu_cuda.py::test_gpu_execution_records_the_requested_index`,
which correctly skips because the repository's primary environment is deliberately CPU-only.

## Reproduction (Linux)

```console
# isolated CUDA 12 environment (does not touch the host driver or system CUDA)
uv venv /tmp/xlaxiom-cuda12 --python 3.12
uv pip install --python /tmp/xlaxiom-cuda12/bin/python "jax[cuda12]==0.11.2" numpy
uv pip install --python /tmp/xlaxiom-cuda12/bin/python --no-deps -e .

# the host LD_LIBRARY_PATH must be cleared for that process only
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  /tmp/xlaxiom-cuda12/bin/python -m xlaxiom doctor

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  /tmp/xlaxiom-cuda12/bin/python -m xlaxiom gpu validate --allow-busy

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 \
  /tmp/xlaxiom-cuda12/bin/python -m xlaxiom gradient-campaign --cases 40 --seed 0
```

Note: invoke via `-m xlaxiom`, not `-m xlaxiom.cli`. The `cli` module has no `__main__`
guard, so `-m xlaxiom.cli` exits 0 without doing anything.

## Safety

No process was killed, no driver was upgraded, no system CUDA installation was altered,
no branch was force-pushed, and no history was rewritten. The P40 (running ComfyUI) and
the RTX 3060 (running VoiceStudio, Chrome, and the desktop compositor) are treated as
busy by default and are only probed behind an explicit `--allow-busy` override that still
requires at least 2048 MiB free VRAM and at most 5% utilization. All GPU UUIDs are
redacted to a stable salted digest; no secrets, private paths, or SSH keys are recorded.

## Artifacts

- `results/gradient-campaign-epyc-cuda.json` — 40-case gradient campaign on `cuda:0`
- `results/doctor-epyc-cuda.json` — `xlaxiom doctor` in the CUDA 12 environment
- `results/matrix-doctor-epyc-cuda.json` — `xlaxiom matrix doctor`
- `results/verification-epyc-cuda.txt` — full gate output
- `results/gpu-validation-epyc-cuda.json` — per-device bounded probe results
- `docs/experiments/epyc-gpu-validation.json` / `.md` — hardware validation evidence
- `docs/experiments/epyc-cuda13-compatibility.json` — empirical CUDA 13 matrix
