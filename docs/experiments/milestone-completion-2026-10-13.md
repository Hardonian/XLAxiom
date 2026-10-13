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

## Release acceptance

The 0.5.0 closure was exercised on every declared Python version (3.11, 3.12,
and 3.13): each run completed 256 tests with two expected no-GPU skips. Ruff lint
and format checks, mypy, lockfile consistency, local Markdown-link validation, and
the source/wheel build also passed.

The wheel was installed into a fresh Python 3.12 environment and invoked outside the
source checkout. Its `--version`, thirteen-property registry, fourteen-entry corpus
validation, and packaged `corpus/curated-v1.json` resource all passed. GitHub reported
no open issues or pull requests at closure time; the pre-closure `main` CI revision was
green. The tag-triggered release workflow independently repeats source gates and the
wheel smoke test, and retains both distributions without claiming package publication.
