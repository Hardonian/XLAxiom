# Changelog

## 0.6.0 - 2026-10-13

- Removed dead code: the unused `DeviceVerdict` value object and `to_json` helper in
  `hardware`, the unused `_materialize`/`_shape`/`api_reshape` helpers in `corpus`, the
  unused `last_error` helper in `gpu_validation`, and the unused
  `expected_hyperbolic_status` table in `historical`. Test-only references were removed
  with them.
- Hardened the isolated version matrix: `MatrixEnvironment` now validates its JAX
  specifier at construction and `load_config` rejects unsafe specifiers before any
  subprocess is spawned, closing an argument-injection path.
- Optimized hot paths: `recommended_variant` selects a CUDA variant in a single pass
  instead of re-enumerating devices per variant, and `_apply` resolves operations from a
  module-level map instead of rebuilding a function dictionary per node.
- Added unit coverage for campaign configuration, isolated execution, memory limits, and
  hardware discovery and capability gating.
- Consolidated the outstanding agent branches into main; all previously open work is
  landed.

## 0.5.3 - 2026-10-13

- Moved the repeat distribution build outside the source checkout so the verification
  output directory cannot affect source-distribution file selection.

## 0.5.2 - 2026-10-13

- Normalized distribution timestamps to the tagged commit time so repeated Linux
  wheel and source-distribution builds are byte-for-byte reproducible.

## 0.5.1 - 2026-10-13

- Added PEP 561 typed-package metadata and modern SPDX license metadata.
- Added explicit Linux CUDA 12 and CUDA 13 installation extras.
- Added three-version installed-wheel verification and Twine metadata checks.
- Added grouped dependency maintenance and guarded PyPI Trusted Publishing automation.
- Added source, wheel, and accelerator installation guidance.

## 0.5.0 - 2026-10-13

- Added failure-preserving minimization to reproduced-finding corpus admission.
- Added identity-checked property comparison across CPU, GPU, and TPU evidence backends.
- Added redacted physical GPU provenance, host/runtime ordinal attribution, and bounded
  JAX GPU-memory behavior to property evidence.
- Retained independent Windows/Linux environment evidence, a 1,300-comparison JAX x64
  campaign, and full thirteen-property HX370/V100/P40 comparisons.
- Added a checked-in curated metamorphic-property corpus with safe closed-registry validation.
- Added bounded property campaigns, coverage reporting, policy overrides, and CI shards.
- Added checksummed relational artifacts with paired minimization, fresh-process replay,
  reports, and bundle execution.
- Added isolated-worker resident-memory limits, cleanup escalation, stability checks, and
  conservative workload-identity performance baseline comparison.
- Added issue-readiness assessment, versioned schema documents, release criteria, support
  matrix, quickstart, and holistic delivery documentation.

## 0.4.0 - 2026-10-12

- Added per-operation, analytic-oracle #40634 historical studies and isolated version execution.
- Added validated execution of portable case bundles without archive extraction.

## 0.3.0 - 2026-10-12

- Added isolated uv-backed JAX version matrix execution and conservative version classification.
- Added checksummed portable ZIP case bundles and explicit CUDA capability discovery.

## 0.2.0 - 2026-10-12

- Enforced JAX device placement and graph-wide symbolic allocation limits.
- Added spawned worker isolation, timeout handling, reusable JIT benchmarking, and failure signatures.
- Added a trusted replay adapter and corpus record for JAX issue #40634.

## 0.1.0 - 2026-10-12

- CPU-first NumPy/JAX differential execution and explicit numerical classification.
- Versioned JSON workloads with deterministic reconstruction and checksums.
- Failure-preserving synthetic control minimization and fresh-process reproduction.
- JSON/Markdown evidence reports, CLI, tests, and CI.
