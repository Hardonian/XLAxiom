# XLAxiom milestone completion audit — 2026-10-13

## Outcome

All fifty software priorities and the four follow-on README engineering tasks are
implemented. Available hardware/platform milestones are backed by retained evidence.
No new JAX/OpenXLA finding was admitted.

## Retained verification

| Scope | Result | Evidence |
| --- | --- | --- |
| Curated corpus | 13 reviewed properties plus 1 labelled historical entry; 0 new findings | [`corpus/findings.json`](../../corpus/findings.json) |
| Longer bounded campaign | 100 seeds, 1,300/1,300 JAX x64 passes, 0 candidates | [`metamorphic-campaign-hx370-x64-100-2026-10-13.json`](metamorphic-campaign-hx370-x64-100-2026-10-13.json) |
| HX370 CPU ↔ V100 | 13/13 workload identities and classifications match | [`metamorphic-crossbackend-hx370-v100-2026-10-13.json`](metamorphic-crossbackend-hx370-v100-2026-10-13.json) |
| V100 ↔ P40 | 13/13 workload identities and classifications match | [`metamorphic-crossdevice-v100-p40-2026-10-13.json`](metamorphic-crossdevice-v100-p40-2026-10-13.json) |
| Independent environments | HX370 Windows and EPYC Linux compare as independent | [`environment-hx370-epyc-comparison-2026-10-13.json`](environment-hx370-epyc-comparison-2026-10-13.json) |
| CUDA/JAX compatibility | CUDA 12/13 behavior retained for V100, P40, and RTX 3060 | [`epyc-cuda13-compatibility.json`](epyc-cuda13-compatibility.json) |

The GPU evidence records redacted physical UUIDs, compute capability, driver/CUDA
versions, host CUDA ordinal, runtime ordinal, and `CUDA_VISIBLE_DEVICES`. JAX device
preallocation is disabled before the first accelerator import so bounded checks do not
claim most of a shared GPU's memory.

## External evidence boundary

TPU eager/JIT property backends are implemented and tested to fail closed without CPU
fallback, but no physical TPU was available. macOS remains a portable but unverified
platform. These are external verification boundaries, not hidden success claims.
