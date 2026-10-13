# Holistic delivery plan

This document maps the fifty-priority program to the repository. “Implemented” means
there is code plus automated coverage. “Prepared” means the repository has a safe,
capability-gated workflow but this host cannot truthfully claim external or physical
verification. It is deliberately not a substitute for hardware evidence.

| # | Priority | Status | Delivery |
| --- | --- | --- | --- |
| 1 | Worker memory enforcement | Implemented | RSS monitoring plus a per-case memory limit. |
| 2 | Worker cleanup escalation | Implemented | Terminate, join, and kill escalation with recorded outcome. |
| 3 | Relational minimization | Implemented | Both workload sides are failure-preservingly minimized. |
| 4 | Relational artifacts | Implemented | Checksummed, data-only paired artifacts. |
| 5 | Fresh-process relational replay | Implemented | `relation reproduce` uses separate interpreters. |
| 6 | Relational reports and bundles | Implemented | JSON/Markdown reports and portable bundle support. |
| 7 | Corpus admission workflow | Implemented | Checksum and reproduction gates reject weak entries. |
| 8 | Independent-environment metadata | Prepared | Required by admission/readiness; a second host must supply it. |
| 9 | Corpus review/status | Implemented | Corpus validation reports composition and evidence status. |
| 10 | Corpus regression CI | Implemented | Property shards and scheduled campaigns run in CI. |
| 11 | More metamorphic relations | Implemented | Eight reviewed declarative relations. |
| 12 | Dtype-specific properties | Partial | `float32` and `int32` covered; x64 remains host-configured evidence. |
| 13 | Non-finite and signed-zero cases | Partial | Comparator and signed-zero relation cover handling; dedicated non-finite relation remains pending. |
| 14 | Randomized property campaigns | Implemented | Bounded deterministic property campaign command. |
| 15 | Property failure triage | Implemented | Pass, candidate violation, and execution/environment triage. |
| 16 | Retain failing seeds/digests | Implemented | Results record seed and both workload SHA-256 values. |
| 17 | Property version matrix | Prepared | Relational artifacts are portable; dedicated isolated relation matrix is pending. |
| 18 | Deterministic CI sharding | Implemented | One CI shard per named property. |
| 19 | Corpus coverage report | Implemented | Operations, dtypes, ranks, and edge-value strategy are reported. |
| 20 | Policy overrides | Implemented | Property CLI accepts explicit `--rtol` and `--atol`. |
| 21 | Campaign profiles | Pending | Named profiles need reviewed resource budgets. |
| 22 | Generator diversity | Partial | Elementwise, broadcast, matmul, reshape, and reduction motifs exist. |
| 23 | Higher-rank/batched matmul | Pending | Requires new bounded generator motifs and evidence. |
| 24 | Integer/mixed dtypes | Partial | `int32` coverage exists; mixed-dtype semantics are not claimed. |
| 25 | Edge-value strategies | Partial | Deterministic signed-zero and magnitude edges exist; configurable strategies are pending. |
| 26 | Overflow/underflow controls | Partial | Historical gradient studies cover overflow; graph controls are pending. |
| 27 | Exception equivalence | Pending | Needs a reviewed exception-comparison policy. |
| 28 | Compilation-cache behavior | Partial | Compilation is timed separately; cache assertions are not evidence-grade. |
| 29 | Nondeterminism detection | Implemented | `xlaxiom stability` records repeated executions and pairwise comparisons. |
| 30 | Performance baselines | Implemented | Benchmark records compare matching workload identities conservatively. |
| 31 | Isolated accelerator campaigns | Implemented | CUDA validation uses isolated interpreters and no CPU fallback. |
| 32 | TPU validation | Prepared | Requested TPU runs fail closed; real validation requires TPU access. |
| 33 | Multi-GPU consistency | Prepared | Identity is recorded; cross-device execution needs idle compatible hardware. |
| 34 | CUDA/JAX compatibility matrix | Partial | CUDA 12/13 selection exists; matrix execution needs target hosts. |
| 35 | GPU provenance | Implemented | Driver, variant, ordinal, mapping, and bounded-memory data are captured. |
| 36 | x64 testing | Prepared | x64 state is captured; x64 CI evidence is required. |
| 37 | Thread/runtime variants | Pending | Needs dedicated isolated-process profiles. |
| 38 | Cross-platform evidence | Prepared | Commands are portable; each platform needs a retained result. |
| 39 | Hardware redaction | Implemented | GPU UUIDs are salted/redacted and process details minimized. |
| 40 | GPU resume/sharding | Pending | Needs a durable campaign ledger on target hardware. |
| 41 | CLI profiles | Pending | See priority 21; profiles must not hide resource changes. |
| 42 | JSON-schema documents | Implemented | Versioned schemas document artifacts and corpus records. |
| 43 | Schema migration strategy | Implemented | Versioning/migration policy is documented; readers reject unknown versions. |
| 44 | Signing/attestation | Prepared | Checksums provide integrity; publisher signing needs release credentials. |
| 45 | Support matrix | Implemented | Support and evidence boundaries are documented. |
| 46 | Scheduled CI | Implemented | Weekly campaign plus manual dispatch. |
| 47 | Reproducible quickstart | Implemented | CPU and capability-gated accelerator steps are documented. |
| 48 | Release criteria | Implemented | Release verification and evidence requirements are documented. |
| 49 | Issue readiness command | Implemented | `xlaxiom readiness` fails closed on missing external evidence. |
| 50 | Documentation index | Implemented | Documentation index links operational evidence surfaces. |

The **Pending** and **Prepared** rows are real boundaries, not deferred claims: closing
them requires new reviewed semantics, physical accelerator access, or release authority.
