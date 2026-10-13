# XLAxiom CUDA hardware validation report

- Platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.43`
- Python: `3.12.13`
- Driver: `580.178.04`
- CUDA: `13.0`

## Summary

- PASS: 3
- UNSUPPORTED: 0
- BLOCKED: 0
- INCONCLUSIVE: 0

## Devices

| CUDA ordinal | nvidia-smi index | Device | SM | Variant | State | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | Tesla V100-SXM2-16GB | 7.0 | cuda12 | PASS | 4 bounded checks passed on Tesla V100-SXM2-16GB under jax[cuda12] |
| 1 | 2 | NVIDIA GeForce RTX 3060 | 8.6 | cuda13 | PASS | 4 bounded checks passed on NVIDIA GeForce RTX 3060 under jax[cuda13] |
| 2 | 1 | Tesla P40 | 6.1 | cuda12 | PASS | 4 bounded checks passed on Tesla P40 under jax[cuda12] |

## Reproduction (Linux)

```console
uv sync --extra dev
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 uv run xlaxiom gpu validate
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 uv run xlaxiom gpu validate
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=2 uv run xlaxiom gpu validate
```

## Notes

- No CPU fallback was used or reported as a GPU result.
- CUDA runtime ordinal order may differ from nvidia-smi index order; both are recorded.
- Host drivers, CUDA installations, and services were not modified.

No secrets, absolute personal paths, SSH keys, or raw GPU UUIDs are recorded.
