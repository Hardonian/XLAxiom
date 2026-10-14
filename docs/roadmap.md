# Holistic delivery plan

This document maps the fifty-priority program to the repository. “Implemented” means
there is code plus automated coverage. “Verified” additionally means retained evidence
exists from the required independent host or physical accelerator. Hardware verification
is deliberately narrower than software completion.

| # | Priority | Status | Delivery |
| --- | --- | --- | --- |
| 1 | Worker memory enforcement | Implemented | RSS monitoring plus a per-case memory limit. |
| 2 | Worker cleanup escalation | Implemented | Terminate, join, and kill escalation with recorded outcome. |
| 3 | Relational minimization | Implemented | Both workload sides are failure-preservingly minimized. |
| 4 | Relational artifacts | Implemented | Checksummed, data-only paired artifacts. |
| 5 | Fresh-process relational replay | Implemented | `relation reproduce` uses separate interpreters. |
| 6 | Relational reports and bundles | Implemented | JSON/Markdown reports and portable bundle support. |
| 7 | Corpus admission workflow | Implemented | Findings are failure-preservingly minimized, reproduced in fresh processes, checksummed, and rejected if any gate fails. |
| 8 | Independent-environment metadata | Verified | Checksummed HX370 Windows and EPYC Linux records establish independent host identity. |
| 9 | Corpus review/status | Implemented | Corpus validation reports composition and evidence status. |
| 10 | Corpus regression CI | Implemented | Property shards and scheduled campaigns run in CI. |
| 11 | More metamorphic relations | Implemented | Thirteen reviewed declarative relations. |
| 12 | Dtype-specific properties | Implemented | `float32`, `int32`, mixed `int32`/`float32`, and explicit-process `float64` relations. |
| 13 | Non-finite and signed-zero cases | Implemented | Dedicated signed-zero and NaN/infinity relations use explicit input modes. |
| 14 | Randomized property campaigns | Verified | The bounded command is covered and a retained 100-seed/1,300-comparison JAX x64 campaign passed. |
| 15 | Property failure triage | Implemented | Pass, candidate violation, and execution/environment triage. |
| 16 | Retain failing seeds/digests | Implemented | Results record seed and both workload SHA-256 values. |
| 17 | Property version matrix | Implemented | `matrix relation` executes checksummed relation artifacts in isolated JAX versions. |
| 18 | Deterministic CI sharding | Implemented | One CI shard per named property. |
| 19 | Corpus coverage report | Implemented | Operations, dtypes, ranks, and edge-value strategy are reported. |
| 20 | Policy overrides | Implemented | Property CLI accepts explicit `--rtol` and `--atol`. |
| 21 | Campaign profiles | Implemented | Smoke, standard, and thorough profiles expose explicit resource budgets. |
| 22 | Generator diversity | Implemented | Elementwise, broadcast, matmul, reshape, reduction, and batched motifs exist. |
| 23 | Higher-rank/batched matmul | Implemented | Rank-three broadcast and batched-matmul motifs are opt-in via generation rank. |
| 24 | Integer/mixed dtypes | Implemented | Integer and mixed add-commutativity relations capture same-operation promotion semantics. |
| 25 | Edge-value strategies | Implemented | Generated workloads select finite, extreme-finite, signed-zero, or non-finite modes. |
| 26 | Overflow/underflow controls | Implemented | Extreme finite normal/subnormal inputs are explicit, bounded, and property-tested. |
| 27 | Exception equivalence | Implemented | `outcomes` compares terminal status categories separately from numerical equality. |
| 28 | Compilation-cache behavior | Implemented | Checksummed cold/warm compilation observations fail closed on mismatched identities. |
| 29 | Nondeterminism detection | Implemented | `xlaxiom stability` records repeated executions and pairwise comparisons. |
| 30 | Performance baselines | Implemented | Benchmark records compare matching workload identities conservatively. |
| 31 | Isolated accelerator campaigns | Implemented | CUDA validation uses isolated interpreters and no CPU fallback. |
| 32 | TPU validation | Verified | Physical TPU v5e (ViperLite) evidence retained under evidence/tpu/; confirms 19/19 metamorphic property equivalence against CPU JIT without CPU fallback. |
| 33 | Multi-GPU consistency | Verified | All 13 checksummed relations match on V100 and P40, with physical identity and placement retained. |
| 34 | CUDA/JAX compatibility matrix | Verified | CUDA 12/13 was physically probed across V100, P40, and RTX 3060; supported and rejected SM combinations are retained. |
| 35 | GPU provenance | Verified | Driver, CUDA variant, host/runtime ordinals, redacted UUID, compute capability, and bounded-memory policy are retained. |
| 36 | x64 testing | Implemented | `--x64` creates a clean JAX x64 interpreter and CI runs every property shard there. |
| 37 | Thread/runtime variants | Implemented | Reviewed single/four-thread fresh-process profiles are executable and recorded. |
| 38 | Cross-platform evidence | Verified | Checksummed Windows HX370 and Linux EPYC records compare as independent; macOS remains portable but unverified. |
| 39 | Hardware redaction | Implemented | GPU UUIDs are salted/redacted and process details minimized. |
| 40 | GPU resume/sharding | Implemented | GPU shards have exact identity-bound, checksummed resumable ledgers. |
| 41 | CLI profiles | Implemented | `campaign --profile` exposes budgets, with explicit flags as overrides. |
| 42 | JSON-schema documents | Implemented | Versioned schemas document artifacts and corpus records. |
| 43 | Schema migration strategy | Implemented | Versioning/migration policy is documented; readers reject unknown versions. |
| 44 | Signing/attestation | Implemented | OpenSSL detached signing/verification is supported; a maintainer key is required to execute it. |
| 45 | Support matrix | Implemented | Support and evidence boundaries are documented. |
| 46 | Scheduled CI | Implemented | Weekly campaign plus manual dispatch. |
| 47 | Reproducible quickstart | Implemented | CPU and capability-gated accelerator steps are documented. |
| 48 | Release criteria | Implemented | Release verification and evidence requirements are documented. |
| 49 | Issue readiness command | Implemented | `xlaxiom readiness` fails closed on missing external evidence. |
| 50 | Documentation index | Implemented | Documentation index links operational evidence surfaces. |

All fifty software priorities are implemented. Retained evidence closes the available
Windows/Linux, V100/P40/RTX 3060, CUDA 12/13, and independent-host milestones. TPU
execution and macOS portability remain explicitly unverified external hardware/platform
claims; the implemented paths fail closed and do not convert CPU execution into evidence.

Key completion artifacts:

- [`environment-hx370-epyc-comparison-2026-10-13.json`](experiments/environment-hx370-epyc-comparison-2026-10-13.json)
- [`metamorphic-campaign-hx370-x64-100-2026-10-13.json`](experiments/metamorphic-campaign-hx370-x64-100-2026-10-13.json)
- [`metamorphic-crossbackend-hx370-v100-2026-10-13.json`](experiments/metamorphic-crossbackend-hx370-v100-2026-10-13.json)
- [`metamorphic-crossdevice-v100-p40-2026-10-13.json`](experiments/metamorphic-crossdevice-v100-p40-2026-10-13.json)
- [`epyc-cuda13-compatibility.json`](experiments/epyc-cuda13-compatibility.json)
