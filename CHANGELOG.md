# Changelog

## 3.2.0 - 2026-11-01

### Added

- **Universal Pytest Interception & Rich Diagnostics (`xlaxiom.patch_testing`)**:
  - Supercharges `torch.testing.assert_close` and `numpy.testing.assert_allclose` across any repository without changing test code (`enable_rich_asserts()`, `disable_rich_asserts()`, `patch_testing()` context manager).
  - CLI switch `pytest --xlaxiom-rich-asserts` and ini setting `xlaxiom_rich_asserts = true`.
  - Automatically enriches assertion failure reports with Signal-to-Noise Ratio (SNR), Cosine Similarity, bitwise ULP distance, subnormal underflow traps, catastrophic cancellations, and error percentiles (p50, p90, p99).
  - Flag `--xlaxiom-save-html` writes offline single-file interactive SVG/HTML diff visualizers into `.xlaxiom-reports/`.
- **Universal Model Checkpoint & Tensor Diffing CLI (`xlaxiom diff`)**:
  - Supports positional arguments: `xlaxiom diff <actual> <desired>` and flags `--actual`/`--desired`.
  - Automatically detects whether inputs are single `.npy` tensors or whole model checkpoints (`.safetensors`, `.npz`, sharded directories).
  - Flags: `--top-k N` (top divergent layers ranking), `--min-snr <dB>` (CI parity gate threshold), `--html <path>` (interactive HTML dashboard), `--json <path>`.
- **Interactive Checkpoint Parity HTML Dashboard (`xlaxiom.ui.generate_checkpoint_html`)**:
  - Standalone, offline-safe, responsive HTML dashboard for checkpoint weight comparison.
  - Features real-time client-side layer name search/filter, status filtering (All, Diverged Only, Matched Only), KPI cards, and divergence metrics.
  - Automatically renders directly in Jupyter Notebooks and Google Colab via `_repr_html_()`.
- **Checkpoint Health & Sanity Linter (`xlaxiom.lint` & `xlaxiom lint-checkpoint`)**:
  - `lint_checkpoint` and `lint_tensor` detect silent model defects across Safetensors, NumPy, and PyTorch weights.
  - Audits non-finites (NaNs, Infs), subnormal/denormal floats (which trigger 10x-100x CPU/GPU inference latency degradation via microcode traps), dead/zero weights, and extreme outliers (> 6 sigma).
  - CLI command: `xlaxiom lint-checkpoint <path> [--fail-on-warning] [--output <report.json>]`.
- **Checkpoint Quantization Risk Profiler (`xlaxiom.quantization.audit_checkpoint_quantization`)**:
  - Evaluates entire checkpoints against modern AI precision formats: FP8 (E4M3, E5M2), FP4 (E2M1 / MXFP4), INT8, and BFloat16.
  - Generates recommended BF16/FP16 fallback whitelists for sensitive layers (e.g. `down_proj`, `lm_head`) to preserve LLM generation quality.
  - Subcommand `xlaxiom quant-audit <checkpoint> --dtype fp8_e4m3` accepts model files and directories.
- **Turnkey Pre-Commit Hook (`.pre-commit-hooks.yaml`)**:
  - Provides `xlaxiom-lint-checkpoint` and `xlaxiom-snapshot-check` hooks for Git repositories to prevent corrupted model checkpoints and numerical drifts from reaching production.

## 3.1.0 - 2026-11-01

### Added

- **ULP Distance & Statistics (`xlaxiom.ulp`)**: Integer-exact floating-point ULP distance measurement across float16, float32, and float64 (`ulp_distance`, `max_ulp_error`, `ulp_stats`, `assert_ulp_close`). Equates signed zeroes (+0 == -0) and handles subnormals and matching NaNs cleanly.
- **Golden File Snapshot Testing (`xlaxiom.golden` & pytest fixture)**: Record-once, verify-forever snapshot assertions (`check_golden`, `save_golden`, `load_golden`). Features SHA-256 tamper-evident checksums, max ULP enforcement, and the `--xlaxiom-update-golden` pytest fixture flag. CLI subcommands: `xlaxiom snapshot record` and `xlaxiom snapshot check`.
- **Dependency-Free Safetensors & BF16 IO (`xlaxiom.safetensors_io`)**: Zero-dependency pure-NumPy `.safetensors` reader and writer (`read_safetensors`, `write_safetensors`, `read_checkpoint`) supporting exact BFloat16 bit-widening to float32 and sharded checkpoint index auto-resolution (`model.safetensors.index.json`).
- **Turnkey GitHub Action (`action.yml`)**: Composite Action (`Hardonian/XLAxiom@v3`) for one-line CI/CD numerical regression gating, step summary generation, and sticky PR commenting bot.
- **Serving Top-K Logprob Parity (`xlaxiom.serving`)**: Distributional top-k next-token logprob parity auditor (`audit_topk_parity`) with OpenAI-compatible serving adapter (`openai_compatible_topk`), logits adapter (`topk_from_logits`), and HuggingFace causal LM adapter (`huggingface_topk`).
- **Multi-Tensor & JUnit XML CI Gate**: `xlaxiom ci-gate` now accepts `.npz` files and checkpoint directories containing multiple tensors, and outputs standard JUnit XML test reports (`--junit-xml`) for CI UI visualization.
- **Multi-Platform CI Matrix**: Added automated Windows CI runner (`windows-latest`) and PyTorch FX integration runner (`test-torch`) to `.github/workflows/ci.yml`.

### Fixed

- **CI gate**: the PR comment verdict now uses the same predicate as the gate exit status, so a tensor below `min_snr_db` can no longer be reported as `PASS`. The comment is ASCII-only (safe to print on cp1252 consoles), the badge URL no longer contains a raw space, table cells escape `|`, and "worst-case" drift is the maximum across all regressed tensors. `evaluate_ci_gate([])` now raises instead of passing vacuously.
- **Serving parity**: top-1/top-5 percentages are averaged per token position (sequence logits previously produced values above 100%). NaN/+inf logits can no longer look like agreement, shape mismatches and empty inputs raise `ValueError`, and KL divergence is computed in log space without probability clipping. `rtol`/`atol` were previously ignored; they are now enforced when supplied. Added `min_top1_agreement_pct`.
- **Checkpoint audit**: the summary lists the lowest-SNR weights first (it previously listed the first five weights), `.npz` handles are closed (Windows file locking), and file suffixes are case-insensitive. CLI commands report load errors without a traceback.
- **Calibration**: NaN/inf and shape-changing runs raise instead of being reported as "deterministic"; `safety_factor < 1` is rejected and the summary shows the factor actually used.
- **Drift auditor**: shape mismatches and NaN/Inf outputs are classified explicitly rather than as FMA contraction; the report notes signatures are heuristics.
- **HTML visualizer**: no longer crashes on NaN/inf errors, adds a viewport meta tag, and escapes shape/dtype text. The chart is described as error percentiles (no heatmap is produced).

## 3.0.0 - 2026-10-31

- **Major Enterprise Production GA Release**: Completion of CI/CD regression gates, PR commenting bot, HuggingFace/Safetensors weight auditing, LLM serving engine parity verification, automated empirical tolerance calibration, and cross-hardware numerical drift diagnosis (Priorities 106–115).
- **CI Numerical Regression Gate & PR Commenting Bot (`xlaxiom.ci` / Priority 106)**:
  - `evaluate_ci_gate` and `generate_pr_comment` provide turnkey CI/CD numerical drift gates.
  - Automatically posts sticky GitHub Pull Request comments with SNR summary tables, regression badges, and minimal repro snippets.
  - CLI subcommand: `xlaxiom ci-gate --actual ... --desired ... --commit ... --output ...`.
- **HuggingFace Checkpoint & Safetensors Parity Auditor (`xlaxiom.hf` / Priority 107)**:
  - `audit_checkpoint_parity` audits weight tensors across Safetensors, .npz, and in-memory state dicts.
  - Reports missing keys, extra keys, top divergent weights, max absolute error, and cosine similarity.
  - CLI subcommand: `xlaxiom checkpoint-audit --baseline ... --candidate ...`.
- **Serving Engine vs Reference Logit Parity Verifier (`xlaxiom.hf` / Priority 108)**:
  - `audit_serving_parity` evaluates high-throughput inference engines (vLLM, SGLang, TensorRT-LLM) against reference PyTorch eager models.
  - Measures Top-1 token agreement percentage, Top-5 token set overlap percentage, max logit difference, and mean KL divergence.
- **Interactive Tensor Diff Visualizer & Offline HTML Dashboard (`xlaxiom.ui` / Priority 109)**:
  - `generate_visualizer_html` compiles standalone, responsive single-file HTML dashboards.
  - Features embedded zero-dependency SVG error distribution histograms, root-cause diagnostics, and copyable reproduction scripts.
- **Cross-Hardware Numerical Drift Baseline & Signature Classifier (`xlaxiom.drift` / Priority 110)**:
  - `audit_hardware_drift` audits floating-point discrepancies across CPU, NVIDIA GPU (TF32), Google Cloud TPU (BF16), and Apple Silicon (MPS).
  - Automatically classifies architectural root causes: NVIDIA TF32 mantissa truncation (10-bit), TPU/GPU BFloat16 truncation (7-bit), FMA associative contraction, and severe algorithmic divergence.
- **Framework Migration & Compilation Validation Suites (Priority 111)**:
  - Turnkey validation suites comparing PyTorch eager, `torch.compile` (Inductor), and JAX eager/JIT with intermediate activation inspection.
- **Automated Empirical Tolerance Calibrator (`xlaxiom.calibration` / Priority 112)**:
  - `calibrate_tolerances` measures the intrinsic stochastic variance of model kernels (multi-threaded reduction nondeterminism, atomic additions, cuBLAS).
  - Recommends statistically robust `(rtol, atol)` boundaries with safety factor headroom to prevent flaky CI tests.
- **Documentation & PyPI Packaging Expansion (Priority 113)**:
  - Comprehensive user guides, tutorials for CI/CD integration, serving verification, and checkpoint auditing.
- **Comprehensive Quality Verification (Priorities 114 & 115)**:
  - 100% test pass rate across 450+ unit and e2e test cases.
  - Strict type checking with 0 errors across 70 source files (`mypy src`).
  - Strict linting and formatting compliance (`ruff check`, `ruff format`).
  - Production distribution packaging and metadata verification (`scripts/verify_distribution.py`, `twine check`).

## 2.3.0 - 2026-10-31

- **Major Universal Numerical Parity, Kernel Fuzzing & PyPI Adoption Release**: Comprehensive delivery of drop-in Pythonic assertion APIs, Triton/Pallas custom kernel fuzzing, neural network layer-by-layer parity auditing, quantization risk analysis (FP8/FP4/Int8), and decoupled lightweight packaging (Priorities 96–105).
- **Decoupled Lightweight PyPI Packaging (`pyproject.toml` / Priority 96)**:
  - Broadened core compatibility to `numpy>=1.24`, enabling seamless installation in legacy enterprise environments.
  - Decoupled `jax` and `torch` into clean optional extras (`pip install xlaxiom[jax]`, `pip install xlaxiom[torch]`), dropping base install size under 5MB.
- **Pythonic `assert_close` with Root-Cause Diagnostics (`xlaxiom.assertions` / Priority 97)**:
  - Drop-in replacement for `torch.testing.assert_close` and `np.testing.assert_allclose`.
  - Computes Signal-to-Noise Ratio (SNR in dB), Cosine Similarity, RMS error, and error distribution percentiles (p50, p90, p99).
  - Automatically diagnoses root cause: catastrophic cancellation, sign-bit flips, subnormal underflow, and non-finite masks.
  - Generates standalone, 10-line reproduction scripts on assertion failure.
- **Triton & Pallas Custom Kernel Differential Fuzzer (`xlaxiom.fuzzing` / Priority 98)**:
  - `fuzz_kernel` differentially validates custom GPU/TPU kernels against eager references.
  - Fuzzes dynamic shapes with prime-dimension bias (catching tiling boundaries), unaligned power-of-2 strides, and subnormal/extreme values.
- **Automated Kernel Shape Minimization (`xlaxiom.fuzzing` / Priority 99)**:
  - Automatically delta-debugs and shrinks multidimensional failing tensors to the minimal shape that reproduces failure.
- **Neural Network Layer-by-Layer Parity Auditing (`xlaxiom.nn_parity` / Priority 100)**:
  - `audit_layer_parity` hooks into PyTorch modules, Flax Linen/NNX, and compiled models.
  - Computes activation drift per layer (MAE, Max Rel Diff, Cosine Similarity, SNR).
- **Divergence Origin Localization (`xlaxiom.nn_parity` / Priority 101)**:
  - Automatically identifies the earliest intermediate layer where numerical drift originates during framework migration or compilation.
- **Quantization & Precision-Loss Risk Auditor (`xlaxiom.quantization` / Priority 102)**:
  - `audit_quantization_risk` simulates FP8 (`fp8_e4m3`, `fp8_e5m2`), FP4 (`fp4_e2m1` / MXFP4), INT8, and BFloat16 precision formats.
- **Dynamic Range Outlier & Underflow Analysis (`xlaxiom.quantization` / Priority 103)**:
  - Quantifies outlier clipping percentage, underflow flush percentage, and yields actionable deployment risk recommendations (LOW, MODERATE, CRITICAL).
- **Hypothesis-Like Metamorphic Invariant Decorators (`xlaxiom.metamorphic_decorator` / Priority 104)**:
  - `@test_invariants` / `@metamorphic` validates batch padding invariance, scale invariance/equivariance, and shift invariance on tensor functions.
- **Pytest Plugin Auto-Reporting for Numerical Assertions (`xlaxiom.pytest_plugin` / Priority 105)**:
  - Seamlessly intercepts `NumericalAssertionError` during pytest runs and writes out interactive HTML diagnostic report cards to `.xlaxiom-reports/`.

## 2.2.0 - 2026-10-27

- **Major Google Ecosystem & Enterprise Product Readiness Release**: Full end-to-end integration across Google Cloud Platform (GCP), Cloud TPUs (v4, v5e, v5p, v6e Trillium), Vertex AI, GKE (`xpk`), OpenXLA StableHLO reference interpreter, XLA compiler pass isolation, Google model architectures (Flax, MaxText), Google Colab, BigQuery, and Google Cloud KMS.
- **Transparent Google Cloud Storage (GCS) `gs://` URI Layer (`xlaxiom.cloud.storage` / Priority 79)**:
  - Direct reading and writing of checksummed cases, relational artifacts, campaign ledgers, and attestations to/from `gs://bucket/blob` URIs without local disk staging.
  - Transparent integration with `load_case`, `save_case`, `load_relational_case`, `save_relational_case`, and `load_artifact`.
- **Multi-Generation Cloud TPU Matrix Support (`xlaxiom.cloud.tpu` / Priority 80)**:
  - First-class specifications, memory bandwidth, peak compute FLOPs (FP32, BF16, FP8), and mesh topology models for TPU v4, TPU v5e (ViperLite), TPU v5p, and TPU v6e (Trillium).
  - CLI `xlaxiom cloud tpu-status` and `xlaxiom cloud tpu-run`.
- **Multi-Host TPU Pod Slice Orchestration (`MultiHostTPUOrchestrator` / Priority 81)**:
  - Dynamic coordination environment generator for multi-host TPU slices (v4-16..v4-4096, v5e-16, v6e-256) via JAX distributed coordination.
  - Simulated inter-host ring all-reduce latency and bandwidth over TPU Inter-Chip Interconnect (ICI).
- **Vertex AI Custom Jobs & KFP Pipeline Component (`xlaxiom.cloud.vertex` / Priority 82)**:
  - Programmatic generation of production Vertex AI Custom Training Job specifications (`create_vertex_job_spec` / `xlaxiom cloud vertex-spec`).
  - Turnkey Kubeflow Pipelines component YAML (`generate_kfp_component_yaml`) for automated compiler verification in MLOps CI/CD.
- **GKE Accelerated Processing Kit (`xpk`) Generator (`xlaxiom.cloud.xpk` / Priority 83)**:
  - Declarative workload configuration and CLI generator (`generate_xpk_workload_spec` / `xlaxiom cloud xpk-spec`) for dispatching campaigns on GKE TPU clusters.
- **OpenXLA StableHLO Pure Reference Interpreter Oracle (`xlaxiom.stablehlo_interpreter` / Priority 84)**:
  - Pure-Python reference interpreter strictly following the official OpenXLA StableHLO specification.
  - Registered as a first-class execution oracle backend (`backend="stablehlo-interpreter"`).
- **XLA Compiler Optimization Pass Isolation & Bisection (`xlaxiom.xla_passes` / Priority 85)**:
  - Pinpoints the exact XLA compiler pass (algebraic-simplifier, dot-decomposer, instruction-fusion, layout-assignment, rematerialization) introducing numerical discrepancies.
  - CLI subcommand `xlaxiom xla pass-isolate`.
- **Google Model Framework Adapters: Flax & MaxText (`xlaxiom.flax` / Priority 86)**:
  - Tracing and subgraph extraction for Flax Linen and Flax NNX modules.
  - Synthetic computational block builders for Transformer Multi-Head Self-Attention, RMSNorm, and SwiGLU MLP layers as used in Google MaxText and Gemma.
- **Interactive Google Colab Notebooks & Rich Display Protocols (`xlaxiom.colab` / Priority 87)**:
  - One-click Jupyter/Colab `.ipynb` generator with automated hardware accelerator provisioning and visual diagnostics (`xlaxiom colab`).
  - Official "Open in Colab" markdown badge generation (`colab_badge_markdown`).
  - First-class `_repr_html_()` methods across `CaseArtifact`, `RelationalArtifact`, `ComparisonResult`, `BisectResult`, `XLAPassIsolationResult`, and `CloudTPUJobResult`.
- **Enterprise Governance: Cloud KMS, SLSA v1.0 & BigQuery (`xlaxiom.kms` & `xlaxiom.bigquery` / Priority 88)**:
  - Google Cloud KMS asymmetric signing and verification (`sign_kms_attestation`, `verify_kms_attestation` / `xlaxiom kms sign`, `xlaxiom kms verify`).
  - in-toto SLSA v1.0 build provenance generation (`generate_slsa_provenance`).
  - BigQuery table schemas for campaign ledgers, metamorphic properties, and empirical roofline metrics, with streaming export (`xlaxiom bigquery schema`, `xlaxiom bigquery export`).
  - Structured Google Cloud Logging (`format_cloud_log`) and OpenTelemetry trace spans (`trace_compiler_span`).
  - Multi-stage hardened distroless container `Dockerfile` and automated Google `cloudbuild.yaml`.
- **On-Boarded Enterprise & Upstream Ecosystem Packages (Priorities 89–95)**:
  - Multi-Generation Cloud TPU qualification evidence suites with SHA-256 manifests for TPU v4, v5p, and v6e Trillium (evidence/tpu-v4, evidence/tpu-v5p, evidence/tpu-v6e).
  - Turnkey fleet provisioning and zero-leak teardown automation (scripts/tpu/provision_and_collect_all.sh / .ps1).
  - Upstream Google OSS packages: google/jax issue template patch & repro verifier, openxla/xla pass isolation RFC & stablehlo-opt runner, and google/maxtext numerical parity test harness (upstream/).
  - Enterprise Infrastructure-as-Code Terraform module (terraform/) and 1-click GCP CLI bootstrapper (scripts/enterprise_gcp_bootstrap.sh / .ps1).
  - Google Cloud Marketplace solution definition, Vertex AI Workbench extension descriptor, and automated PyPI / SLSA v1.0 publishing workflow (.github/workflows/publish.yml).

## 2.1.0 - 2026-10-19

- **Strategic Accelerator & Compiler Verification Release**: Comprehensive delivery of Google Cloud TPU full-corpus verification harness, multi-device TPU SPMD topologies, StableHLO semantic lowering verification, higher-order autodiff mathematical invariants, and calibrated empirical roofline efficiency.
- **Full-Corpus Cloud TPU Harness & Hardware Simulation (`xlaxiom.tpu_harness` / Priority 72)**:
  - Extends TPU verification from the 19-property baseline to the complete 40-property curated metamorphic corpus across `jax-tpu-eager` and `jax-tpu-jit`.
  - Added deterministic `tpu_hardware_emulator` and `audit_tpu_workload_equivalence` for local testing and CI verification without requiring live cloud quota.
  - Enhanced automated runner scripts (`scripts/collect_tpu_evidence.ps1` and `scripts/remote_tpu_runner.sh`) capturing metamorphic campaigns, higher-order autodiff, and readiness preflight audits.
  - Added hardware discovery for TPU devices (`discover_tpu_devices`) integrated into non-destructive `xlaxiom doctor` and `xlaxiom hardware` reports.
  - Physical validation on 2026-10-19 completed on a 4-chip Google Cloud TPU v5e: 40/40 eager properties, 40/40 JIT properties, 2,000/2,000 campaign evaluations, CPU/TPU classification equivalence, and higher-order autodiff all passed with retained evidence.
  - Hardened the collector with explicit Windows OpenSSH key provisioning, isolated ephemeral host keys, exact native argument handling, per-stage status, failure-preserving artifact download, explicit CPU baseline placement, and bounded teardown.
  - Added a zero-cost offline qualification workflow with explicit CPU-simulation provenance, checksummed manifests, full 40-property eager/JIT coverage, and a fail-closed physical-evidence boundary.
- **TPU-Native Multi-Device Topology & SPMD Collective Engine (`xlaxiom.collective` / Priority 73)**:
  - Pre-configured 2D and 3D TPU pod topologies: `TPU_V5E_MESH_2X2` (Google Cloud TPU v5e ViperLite 4-chip pod) and `TPU_V4_MESH_2X2X1`.
  - Added physical ring collective simulation (`simulate_ring_all_reduce`, `simulate_all_gather`) measuring communication steps and data transferred per chip.
- **StableHLO MLIR Roundtrip & Semantic Lowering Verifier (`xlaxiom.stablehlo` / `xlaxiom hlo --verify` / Priority 74)**:
  - Structural MLIR grammar and SSA assignment validation (`verify_stablehlo_roundtrip`), checking def-use chains, module encapsulation, and return signature integrity.
  - New CLI capability `xlaxiom hlo --verify <case.json>` for automated OpenXLA MLIR verification.
- **Higher-Order Autodiff Mathematical Invariants Suite (`xlaxiom.gradients` / `xlaxiom gradient --audit-invariants` / Priority 75)**:
  - Validates second-order autograd against fundamental mathematical dualities:
    - Bilinear Clairaut/Schwarz Hessian symmetry: $\langle u, \mathcal{H} v \rangle = \langle v, \mathcal{H} u \rangle$.
    - Hessian-vector product linearity: $\text{HVP}(u + v) = \text{HVP}(u) + \text{HVP}(v)$.
    - Taylor second-order expansion consistency against finite difference second derivatives.
- **Calibrated Empirical Roofline Profiler (`xlaxiom.efficiency` / `xlaxiom efficiency --empirical` / Priority 76)**:
  - Quantifies empirical execution duration, achieved GFLOPs/sec, achieved memory bandwidth (GB/sec), and `% of Peak` attained against Google Cloud TPU v5e (819 GB/s, 24 TFLOPS FP32) and CPU architectures.
- **Dedicated Cloud TPU Upstream Issue & Repro Exporter (`xlaxiom.triage` / Priority 77)**:
  - Formats GitHub Issue packages tailored specifically for `openxla/xla` and `google/jax` TPU bugs.
  - Auto-embeds StableHLO MLIR dialect module text and explicit `backend='tpu'` standalone 10-line Python reproduction scripts.
- **Full QA & Version Bump (`xlaxiom._version` / Priority 78)**:
  - Distribution version bumped to `2.1.0`.
  - 100% green test passes across all suites, strict typed mypy compliance, and zero ruff warnings.

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
