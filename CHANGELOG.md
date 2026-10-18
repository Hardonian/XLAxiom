# Changelog

## 2.0.0 - 2026-10-17

- **Major v2.0.0 Strategic Compiler Verification Release**: Comprehensive delivery of next-generation compiler verification, dialect interoperability, distributed SPMD simulation, and higher-order automatic differentiation.
- **Dynamic Shapes & Symbolic Dimensions Engine (`xlaxiom.symbolic` / Priority 62)**:
  - First-class symbolic dimension abstractions (`SymbolicDim`, `DynamicInputSpec`) enabling compilation testing across parametric batch sizes, variable sequence lengths, and dynamic rank boundaries.
  - Rigorous symbolic broadcast inference (`infer_symbolic_broadcast`) and concrete instance resolution (`instantiate_workload`) catching polyhedral and loop-bound compiler bugs.
- **Pure Compiler IR Dialect Exporter: StableHLO / MLIR (`xlaxiom.stablehlo` / `xlaxiom hlo` / Priority 63)**:
  - Direct, pure Python translation from XLAxiom `Workload` DAGs to canonical StableHLO MLIR module syntax (`@xlaxiom { func.func @main(...) -> ... }`).
  - Native support for elementwise arithmetic, transcendentals, `stablehlo.dot_general`, `stablehlo.reshape`, and `stablehlo.reduce`.
  - Enables zero-dependency integration into `stablehlo-opt`, `stablehlo-translate`, and upstream OpenXLA test suites.
- **Higher-Order Automatic Differentiation Engine (`xlaxiom.gradients` / Priority 64)**:
  - Forward-mode Jacobian-vector products (`compute_jvp`) and reverse-mode vector-Jacobian products (`compute_vjp`).
  - Forward-over-reverse Hessian-vector products (`compute_hvp`) validated against analytic second-derivative references (`analytic_hessian`) and finite-difference triangulation.
  - New CLI capability `xlaxiom gradient --higher-order` for automated second-order AD parity audits.
- **Arithmetic Intensity & Roofline Efficiency Profiler (`xlaxiom.efficiency` / `xlaxiom efficiency` / Priority 65)**:
  - Quantitative operational FLOP counting and minimal theoretical memory traffic accounting (FLOPs/Byte).
  - Multi-target roofline classification against standard hardware profiles (NVIDIA V100/A100/H100, Google TPU v5e, and x86-64 AVX-512 CPU).
  - Pinpoints memory-bound vs compute-bound operational regimes and calculates achievable peak GFLOPs.
- **Multi-Device SPMD & Sharding Consistency Simulator (`xlaxiom.collective` / Priority 66)**:
  - Single-host simulation of multi-dimensional logical device topologies (`DeviceMesh`) and tensor layout partitions (`PartitionSpec`).
  - Collective reduction communication simulation (`simulate_all_reduce`) verifying distributed sharded computation against replicated single-device ground truth.
- **PyTorch FX Graph Adapter (`xlaxiom.fx` / Priority 67)**:
  - Bidirectional bridge between PyTorch 2.0 FX representations (`torch.fx.GraphModule`) and XLAxiom declarative `Workload` IR.
  - Ingests `torch.compile` / `torch.export` graphs directly for metamorphic differential verification without manual bridges.
- **Interactive Offline HTML Report 2.0 with Embedded SVG DAG Visualizer (`xlaxiom.reporting` / Priority 68)**:
  - Generates self-contained, interactive SVG computation DAG visualizers embedded directly into offline HTML evidence reports.
  - Features topological layer placement, color-coded node categories (inputs, operations, outputs), and responsive layout with zero external JavaScript dependencies.
- **Automated Upstream Issue Readiness Pre-Flight Validator (`xlaxiom readiness --preflight` / Priority 69)**:
  - Automated pre-submission audit validating artifact integrity, non-synthetic status, self-contained standalone reproduction scripts, and redaction of local host user paths and credentials.
- **Expanded Metamorphic Corpus (40 Curated Invariants / Priority 70)**:
  - Added five new reviewed algebraic and lattice invariants:
    - Min-Max De Morgan negation duality (`min-max-demorgan-v1`): $\min(x, y) = -\max(-x, -y)$
    - Hyperbolic tangent odd symmetry (`tanh-odd-symmetry-v1`): $\tanh(-x) = -\tanh(x)$
    - Sine odd function symmetry (`sin-odd-symmetry-v1`): $\sin(-x) = -\sin(x)$
    - Absolute distance subtraction symmetry (`abs-subtraction-symmetry-v1`): $|x - y| = |y - x|$
    - Lattice min-max sandwich absorption (`maximum-minimum-sandwich-v1`): $\min(x, \max(x, y)) = x$
  - 100% verified mutation sharpness and zero-discrepancy agreement across backends.
- **Full QA & Version Bump (`xlaxiom._version` / Priority 71)**:
  - Distribution version bumped to `2.0.0`.
  - 100% green test passes across all suites, zero ruff violations, and strict typed mypy compliance.

## 1.0.0 - 2026-10-16

- **General Availability & Full Release Milestone**: Complete delivery of all strategic compiler evidence capabilities, turning XLAxiom into an essential, universal developer tool across frameworks and accelerators.
- **Graph Tracing & Code Ingestion (`xlaxiom.trace` / `xlaxiom trace`)**:
  - Direct translation of live JAX callables (`jax.make_jaxpr`) into safe, declarative XLAxiom `Workload` IR.
  - Subgraph extraction engine (`extract_subgraph`) that isolates minimal dependency closures around observed numerical discrepancies.
  - Live verification decorator (`@xlaxiom.capture`) for inline differential assertions across execution backends.
  - New CLI subcommand `xlaxiom trace` for ingesting arbitrary Python functions with explicit tensor shapes and dtypes.
- **Automated Compiler Version Bisection (`xlaxiom bisect`)**:
  - Automated binary search across compiler releases using isolated `uv` virtual environments (`MatrixEnvironment`).
  - Automatically isolates the exact compiler release or commit introducing a numerical divergence or regression.
  - Emits structured bisection ledgers and copy-pasteable reproduction commands.
- **Automated Upstream Triage Exporter (`xlaxiom triage`)**:
  - One-click GitHub Issue generator generating audit-ready markdown packages formatted for `google/jax`, `pytorch/pytorch`, and `openxla/xla` issue templates.
  - Produces complete numerical error distribution tables (Max Abs, Max Rel, MAE, RMS), environment provenance, and a standalone 10-line zero-dependency Python reproduction script.
- **Expanded Metamorphic Corpus (from 24 to 35 invariants)**:
  - Added eleven new reviewed algebraic and lattice invariants:
    - Absolute value negation involution (`abs-negation-involution-v1`)
    - Maximum and minimum idempotence (`maximum-idempotence-v1`)
    - Lattice min-max absorption identity (`lattice-absorption-v1`)
    - Absolute value product distributivity (`abs-multiplication-v1`)
    - Square root of square identity (`sqrt-square-identity-v1`)
    - Global mean reduction negation duality (`mean-negation-duality-v1`)
    - Multiplication distributivity over addition (`distributivity-mul-add-v1`)
    - Addition associativity (`add-associativity-v1`)
    - Multiplication associativity (`mul-associativity-v1`)
    - Square absolute self-multiplication identity (`square-abs-multiplication-identity-v1`)
    - Cosine even function symmetry (`cos-even-duality-v1`)
  - Evaluated mutation sharpness across all new properties, confirming 60%–100% mutant kill rates.
- **Developer Ergonomics & Testing Framework (`pytest-xlaxiom`)**:
  - First-class `pytest` plugin registered via entry-point `pytest11`.
  - Provides custom markers (`@pytest.mark.xlaxiom`, `@pytest.mark.xlaxiom_metamorphic`) and fixtures (`xlaxiom_workload_factory`, `xlaxiom_evidence_dir`).
  - Automatic test failure hook emitting offline HTML evidence reports in `.xlaxiom-reports/`.
- **Multi-Framework & Accelerator Expansion**:
  - Enhanced PyTorch adapter (`execute_torch`) with support for TorchInductor compilation (`mode="compile"`) and Apple Silicon Metal/MPS acceleration (`device="mps"`).
  - Added hardware discovery for Apple Silicon (`discover_apple_silicon`) and AMD ROCm (`discover_rocm_devices`) integrated into non-destructive hardware inventory reports.
- **Official GitHub Action (`.github/actions/xlaxiom-action/action.yml`)**:
  - Composite action enabling drop-in CI verification with automated `$GITHUB_STEP_SUMMARY` report tables and artifact uploads.

## 0.6.2 - 2026-10-14

- Expanded the reviewed metamorphic corpus from 19 to 24 declarative invariants, adding
  multiplication commutativity (`mul-commutativity-v1`), minimum/maximum sum identity
  (`min-max-addition-identity-v1`), square of absolute value identity (`square-absolute-identity-v1`),
  multiplication negation duality (`negation-multiplication-duality-v1`), and global sum reduction
  negation duality (`sum-negation-duality-v1`).
- Evaluated mutation sharpness across all new metamorphic invariants (`xlaxiom property sharpness`),
  confirming mutant kill rates between 67% and 100% against synthetic compiler fault hypotheses.
- Modernized the single-file offline HTML evidence report (`build_html_report`) with:
  - System-responsive Dark and Light mode themes with an interactive toggle button and persistent local storage preference.
  - Interactive one-click reproduction command copy button with visual feedback.
  - Collapsible raw JSON evidence inspector card for transparent auditability.
- Added Darwin macOS memory sampling fallback using POSIX `resource.getrusage(RUSAGE_SELF)` and `ps -o rss=` in `safety.py`.
- Hardened CI macOS diagnostics capture and stabilized micro-benchmark comparative timing thresholds against OS scheduling jitter.
- Executed and retained a 10-case / 240-evaluation JAX JIT x64 campaign artifact (`experiments/metamorphic-campaign-v1-24props.json`) confirming 100% pass across all 24 metamorphic properties.

## 0.6.1 - 2026-10-14

- Conducted physical Google Cloud TPU v5e (ViperLite, 4-chip `v5litepod-4`) execution and
  retained concrete verification artifacts (`evidence/tpu/`) confirming 19/19 metamorphic property
  equivalence against CPU JIT and passing a 25-case random campaign without CPU fallback.
- Expanded the safe operation registry with eight new numerical operations: `maximum`,
  `minimum`, `sqrt`, `square`, `log`, `prod`, `max`, and `min`.
- Expanded the curated metamorphic corpus from 13 to 19 declarative invariants, adding
  maximum/minimum commutativity, min-max duality, square as self-multiplication, square
  negation involution, and min-max global reduction duality.
- Implemented offline single-file interactive HTML evidence reporting (`build_html_report`,
  `write_html_report`, `command_report`) featuring responsive metric cards, inline SVG
  tolerance gauges, and environment provenance tables.
- Added a full mutation testing and metamorphic property sharpness engine
  (`xlaxiom property sharpness`, `src/xlaxiom/mutation.py`) that systematically injects
  compiler fault hypotheses to evaluate test suite sensitivity.
- Added deterministic boundary suite synthesis (`generate_boundary_suite` in `generation.py`)
  probing workloads across zeros, ones, finite edges, extreme subnormals, signed zeros,
  and non-finite values.
- Added an optional fail-closed PyTorch differential execution adapter (`execute_torch` in
  `execution.py`) for cross-framework verification against Torch / Inductor.
- Added targeted operation generation in `GenerationConfig` via `preferred_op`, allowing
  focused fuzzing and verification of specific operations across unary, binary, and reduction
  classes.
- Added comprehensive error distribution metrics to `OutputMetrics` in numerical comparison:
  `mean_absolute_error` and `rms_error` (root-mean-square error) are now computed alongside
  the worst-case maximum errors.
- Enhanced `benchmark_jax` with detailed percentiles (`min_ms`, `max_ms`, `p90_ms`, `p95_ms`)
  and attached the workload SHA-256 digest to benchmark artifacts for seamless baseline
  comparisons.
- Added Apple Silicon (Darwin `macos-latest`) test verification job in the GitHub Actions CI matrix
  with retained diagnostic artifacts (`doctor.json`, `property-coverage.json`).
- Enriched `xlaxiom doctor` diagnostics with runtime NumPy version detection.
- Added end-to-end integration coverage for the `benchmark` and `cache` CLI workflows.

## 0.6.0 - 2026-10-13

- Added a shared, positional execution plan that resolves graph names, parameters, and
  operation lookups once per run; NumPy and JAX now share it, and a validated
  NumPy-compatible adapter is available for interoperable eager execution.
- Corrected integer comparisons at signed extrema. `int32`/`int64` minimum values no
  longer overflow during equality checks, and the full `int64` range cannot wrap into
  a false pass under a small tolerance.
- Made validation account for NumPy dtype promotion when enforcing intermediate-memory
  limits, so mixed-dtype operations cannot understate their allocation footprint.
- Bundled the known JAX #40634 historical record and load it through package resources,
  allowing `corpus build` to work from an installed wheel or any current directory.
- Reframed the README and documentation entry points around complete user journeys;
  moved host-specific evidence detail out of the primary product surface.
- Corrected the documented property-comparison workflow so it creates both evidence
  artifacts before attempting a comparison.
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
- Raised coverage on the two modules carrying the core correctness claims: the spawned
  worker boundary (3 tests to 24, 57% to 81%) and the relational artifact pipeline
  (55% to 100%). Every classified worker exit path and every relational failure mode is
  now exercised.
- Corrected the relative-error metric, which previously divided by an `atol` floor and so
  reported a 10,000% relative error for a 1e-7 absolute difference. It now scales by each
  element's own reference magnitude. The discrepancy decision itself is unchanged and is
  pinned by test.
- Validated the benchmark measurement budget: `benchmark_jax(measurements=0)` previously
  raised an opaque `ZeroDivisionError` and now raises a clear `ValueError`.

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
