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
| 11 | More metamorphic relations | Implemented | Twenty-four reviewed declarative relations spanning algebraic commutativity, min-max duality and addition identities, square-absolute duality, and reduction negation. |
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
| 32 | TPU validation | Verified | Physical 4-chip TPU v5e evidence retained under `evidence/tpu/`; confirms 40/40 eager, 40/40 JIT, 2,000/2,000 campaign evaluations, CPU/TPU classification equivalence, x64 execution, and higher-order autodiff. |
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
| 51 | Live graph tracing & code ingestion | Implemented | `xlaxiom.trace`, `extract_subgraph`, `@capture`, and CLI `trace` ingest arbitrary Python/JAX callables. |
| 52 | Subgraph isolation & dependency minimization | Implemented | `extract_subgraph` extracts minimal dependency closures around suspect numerical operations. |
| 53 | Automated compiler version bisection | Implemented | `xlaxiom bisect` binary searches across compiler releases in isolated `uv` virtual environments. |
| 54 | Upstream issue triage & repro exporter | Implemented | `xlaxiom triage` generates 10-line standalone repro scripts and GitHub issue markdown packages. |
| 55 | Metamorphic corpus expansion (35 invariants) | Implemented | Expanded from 24 to 35 invariants covering algebraic and lattice dualities with verified sharpness. |
| 56 | Multi-compiler support: PyTorch Inductor | Implemented | `execute_torch` supports TorchInductor compilation (`mode="compile"`) and MPS device execution. |
| 57 | Apple Silicon MPS discovery & support | Implemented | `discover_apple_silicon` discovers Metal device specifications without synthetic fallback. |
| 58 | AMD ROCm / HIP device discovery | Implemented | `discover_rocm_devices` inventories AMD ROCm GPUs via `rocm-smi` and sysfs. |
| 59 | Developer ergonomics: pytest-xlaxiom | Implemented | First-class `pytest` plugin with markers (`@pytest.mark.xlaxiom`), fixtures, and HTML failure reports. |
| 60 | Official CI Action: xlaxiom-action | Implemented | Turnkey GitHub Action (`.github/actions/xlaxiom-action/action.yml`) with automated step summaries. |
| 61 | Full release test & type verification | Implemented | 100% test pass rate across 378+ tests, zero ruff violations, and clean strict `mypy src`. |
| 62 | Dynamic shapes & symbolic dimensions | Implemented | `SymbolicDim`, `DynamicInputSpec`, dynamic bounds checking, and symbolic broadcast inference. |
| 63 | StableHLO / MLIR text dialect exporter | Implemented | Pure `export_stablehlo` translates workloads directly to MLIR text format without external compiler bindings; CLI `xlaxiom hlo`. |
| 64 | Higher-order automatic differentiation | Implemented | JVP, VJP, HVP forward-over-reverse autodiff, and analytic Hessian validation; CLI `xlaxiom gradient --higher-order`. |
| 65 | Arithmetic intensity & roofline profiler | Implemented | Quantifies FLOPs, memory transfer, operational intensity (FLOPs/Byte), and roofline regimes (V100/A100/H100/TPU/CPU); CLI `xlaxiom efficiency`. |
| 66 | Multi-device SPMD sharding simulator | Implemented | Simulates `DeviceMesh`, `PartitionSpec`, and AllReduce collectives, verifying replicated vs sharded numerical equivalence. |
| 67 | PyTorch FX graph adapter | Implemented | Bidirectional `workload_from_fx` and `workload_to_fx` bridges PyTorch 2.0 / `torch.compile` graphs into XLAxiom IR. |
| 68 | Interactive HTML Report 2.0 with SVG DAG | Implemented | Embeds zero-dependency, responsive SVG computation DAG visualizers directly into single-file offline HTML reports. |
| 69 | Issue readiness pre-flight validator | Implemented | `xlaxiom readiness --preflight` audits cases for self-containment, privacy/path sanitization, and minimal reproduction scripts. |
| 70 | Metamorphic corpus expansion (40 invariants) | Implemented | Expanded to 40 curated invariants with lattice dualities, metric distance symmetries, and verified mutation sharpness. |
| 71 | Version 2.0.0 Release & Full QA | Implemented | Version 2.0.0 GA delivery with 100% green test passes across all suites, strict mypy typing, and clean ruff compliance. |
| 72 | Full-corpus TPU verification & simulation harness | Verified | Complete 40-property physical verification across `jax-tpu-eager` / `jax-tpu-jit`, a 50-case campaign, hardened live runners, retained evidence, and deterministic `TPUHardwareEmulator`. |
| 73 | TPU-native multi-device topology & SPMD collectives | Implemented | First-class `TPU_V5E_MESH_2X2` / `TPU_V4_MESH_2X2X1`, ring AllReduce, and AllGather collective simulation. |
| 74 | StableHLO MLIR roundtrip & semantic lowering verifier | Implemented | `verify_stablehlo_roundtrip` verifies structural MLIR grammar, SSA def-use chains, and semantic lowering parity; CLI `xlaxiom hlo --verify`. |
| 75 | Higher-order autodiff mathematical invariants | Implemented | Bilinear Clairaut/Schwarz Hessian symmetry ($\nabla^2 f = (\nabla^2 f)^T$), Hessian linearity, and Taylor consistency; CLI `xlaxiom gradient --audit-invariants`. |
| 76 | Calibrated empirical roofline profiler | Implemented | `profile_efficiency_empirical` measures live kernel latency, achieved compute throughput (GFLOPs), memory bandwidth (GB/s), and `% of Peak` for TPU v5e; CLI `xlaxiom efficiency --empirical`. |
| 77 | Dedicated Cloud TPU upstream issue exporter | Implemented | `format_triage_issue(..., target_tracker="tpu")` auto-bundles StableHLO MLIR dialect text, TPU device topology, and 10-line standalone repro scripts. |
| 78 | Version 2.1.0 Strategic Release & Full QA | Implemented | Version 2.1.0 delivery with 100% green test passes across all suites, strict mypy typing, clean ruff compliance, and updated evidence suites. |
| 79 | Google Cloud Storage (GCS) `gs://` URI layer | Implemented | Native `gs://bucket/blob` reading/writing for cases, relational artifacts, ledgers, and bundles. |
| 80 | Multi-Generation Cloud TPU Matrix | Implemented | Native specifications and mesh topology models for TPU v4, v5e (ViperLite), v5p, and v6e (Trillium). |
| 81 | Multi-Host TPU Pod Slice Coordination | Implemented | `MultiHostTPUOrchestrator` generates JAX distributed coordination environments and simulates ICI ring all-reduce. |
| 82 | Vertex AI Custom Jobs & KFP Pipeline Component | Implemented | `create_vertex_job_spec` and `generate_kfp_component_yaml` for automated Vertex AI bisection and differential pipelines. |
| 83 | GKE Accelerated Processing Kit (`xpk`) Generator | Implemented | `generate_xpk_workload_spec` formats turnkey GKE TPU cluster dispatch commands. |
| 84 | OpenXLA StableHLO Pure Reference Interpreter | Implemented | `interpret_stablehlo` executes workloads according to formal OpenXLA StableHLO specification semantics. |
| 85 | XLA Compiler Optimization Pass Isolation | Implemented | `isolate_xla_compiler_pass` pinpoints the exact optimization pass (fusion, simplifier, layout) causing numerical divergence. |
| 86 | Google Model Framework Adapters (Flax & MaxText) | Implemented | Ingestion and synthetic extraction of MaxText Multi-Head Attention, RMSNorm, and SwiGLU computational blocks. |
| 87 | Interactive Google Colab Notebooks & Rich Display | Implemented | One-click Colab `.ipynb` generator, official Colab badges, and embedded SVG DAG `_repr_html_()` rendering. |
| 88 | Enterprise Governance: Cloud KMS, SLSA & BigQuery | Implemented | Google Cloud KMS asymmetric signing, in-toto SLSA v1.0 provenance, BigQuery regression schemas, and Cloud Logging. |
| 89 | Multi-Generation Cloud TPU Evidence Suites | Implemented | Disclosed offline qualification packages with topology and SHA-256 manifests for TPU v4, v5p, and v6e Trillium. |
| 90 | Cloud TPU Fleet Automated Provision & Teardown | Implemented | Zero-leak automation scripts (scripts/tpu/provision_and_collect_all.sh / .ps1) for multi-generation TPU VMs. |
| 91 | Upstream JAX Issue Template Integration Package | Implemented | Standalone 10-line repro template patch, PR proposal, and verifier for google/jax (upstream/jax/). |
| 92 | Upstream OpenXLA Compiler Pass Isolation RFC | Implemented | RFC and subprocess stablehlo-opt runner for OpenXLA working group triage (upstream/openxla/). |
| 93 | Upstream MaxText Layer Differential Parity Harness | Implemented | Turnkey test harness for google/maxtext verifying Attention, RMSNorm, and SwiGLU parity (upstream/maxtext/). |
| 94 | Enterprise Infrastructure-as-Code Terraform Module | Implemented | Full Terraform module (terraform/) and GCP bootstrap CLI (scripts/enterprise_gcp_bootstrap.sh) for KMS, BigQuery, and GCS. |
| 95 | Google Cloud Marketplace & Release Workflow | Implemented | Marketplace solution definition (marketplace/), Vertex AI Workbench extension, and PyPI Trusted Publishing with SLSA. |
| 96 | Zero-Friction PyPI Packaging & Decoupled Dependencies | Implemented | Broad `numpy>=1.24` compatibility, decoupled optional extras (`xlaxiom[jax]`, `xlaxiom[torch]`), and lightweight <5MB install. |
| 97 | Pythonic `assert_close` with Root-Cause Diagnostics & SNR | Implemented | Drop-in assertion engine computing SNR (dB), cosine similarity, catastrophic cancellation, sign flips, subnormal underflow, and error percentiles. |
| 98 | Triton & Pallas Custom Kernel Differential Fuzzer | Implemented | `fuzz_kernel` searches edge-case boundary conditions across prime dimensions, power-of-2 strides, subnormals, and non-finite masks. |
| 99 | Automated Shape Minimization for Kernel Boundary Failures | Implemented | Delta-debugging engine shrinks multidimensional failing tensors to minimal reproducing shapes and generates 10-line standalone repro scripts. |
| 100 | Model & Layer-by-Layer Activation Parity Auditor | Implemented | `audit_layer_parity` hooks into PyTorch modules, Flax Linen/NNX, and compiled pipelines to extract intermediate activations. |
| 101 | Divergence Origin Localization for Cross-Framework Migration | Implemented | Automatically pinpoints the exact computational layer where numerical drift originates during eager-to-compiled or PyTorch-to-JAX migration. |
| 102 | Quantization & Precision-Loss Risk Auditor (FP8, FP4, Int8) | Implemented | `audit_quantization_risk` simulates FP8 (E4M3, E5M2), FP4 (E2M1 / MXFP4), and INT8 dynamic range and quantization noise. |
| 103 | Dynamic Range Outlier & Underflow Analysis Engine | Implemented | Quantifies outlier clipping percentage, underflow flush percentage, and outputs actionable precision recommendations. |
| 104 | Hypothesis-Like Metamorphic Invariant Decorator | Implemented | `@test_invariants` / `@metamorphic` validates batch padding invariance, scale invariance/equivariance, and shift invariance on tensor functions. |
| 105 | Pytest Plugin Auto-Reporting for Numerical Assertions | Implemented | Seamless pytest plugin intercepting `NumericalAssertionError` and dumping interactive HTML diagnostic reports to `.xlaxiom-reports/`. |
| 106 | Continuous Integration Regression Gate & PR Commenting Bot | Implemented | `evaluate_ci_gate`, `generate_pr_comment`, and CLI `ci-gate` for automated PR commenting, regression prevention, and SNR quality gates. |
| 107 | HuggingFace Checkpoint & Safetensors Weight Parity Auditor | Implemented | `audit_checkpoint_parity` and CLI `checkpoint-audit` compare safetensors, .npz, and state dict weights with missing/extra key alerts. |
| 108 | Serving Engine vs Reference Logit Parity Verifier | Implemented | `audit_serving_parity` evaluates vLLM, SGLang, and TensorRT-LLM logit fidelity, Top-1/Top-5 token agreement, and KL divergence. |
| 109 | Interactive Tensor Diff Visualizer & Offline HTML Dashboard | Implemented | `generate_visualizer_html` produces standalone, responsive single-file dashboards with SVG error histograms and repro snippets. |
| 110 | Cross-Hardware Numerical Drift Baseline & Signature Classifier | Implemented | `audit_hardware_drift` classifies root architectural causes (TF32 mantissa truncation, BF16 rounding, FMA contraction) across devices. |
| 111 | Framework Migration & Compilation Validation Suites | Implemented | End-to-end parity assertions and layer auditing across PyTorch eager, torch.compile Inductor, and JAX eager/JIT. |
| 112 | Automated Empirical Tolerance Calibrator | Implemented | `calibrate_tolerances` measures kernel stochastic variance to recommend statistically robust (rtol, atol) settings and eliminate flaky CI tests. |
| 113 | PyPI Packaging & Documentation Expansion | Implemented | Version 3.0.0 documentation, quickstarts, decoupled extras, and comprehensive user guides. |
| 114 | Comprehensive Verification, Type Checking, and Linting | Implemented | 100% test pass rate across 450+ unit and e2e tests, strict mypy typing (0 errors across 70 source files), and clean ruff compliance. |
| 115 | Production Distribution Build & Release Packaging | Implemented | Clean wheel and sdist build, metadata validation with twine, and release distribution verification. |

All one hundred and fifteen software, ecosystem, and PyPI adoption priorities are implemented for the **v3.0.0 Production GA Release**.
Retained evidence closes the available Windows/Linux, V100/P40/RTX 3060, CUDA 12/13, independent-host, and Google Cloud TPU v5e
milestones. Apple Silicon and AMD ROCm discovery and execution paths are implemented with strict fail-closed semantics to ensure
no silent CPU fallback occurs.

The 2026-10-19 physical TPU v5e rerun passed all 40 eager properties, all 40 JIT properties, and 2,000/2,000 campaign evaluations
with JAX x64 enabled. The retained CPU baseline and TPU evidence match classifications for all 40 checksummed workloads, and the
higher-order autodiff check passed. The zero-cost offline qualification remains available for orchestration and evidence-integrity
testing when physical TPU access is unavailable, with explicit CPU-simulation provenance.


Key completion artifacts:

- [`environment-hx370-epyc-comparison-2026-10-13.json`](experiments/environment-hx370-epyc-comparison-2026-10-13.json)
- [`metamorphic-campaign-hx370-x64-100-2026-10-13.json`](experiments/metamorphic-campaign-hx370-x64-100-2026-10-13.json)
- [`metamorphic-campaign-v1-24props.json`](experiments/metamorphic-campaign-v1-24props.json)
- [`metamorphic-crossbackend-hx370-v100-2026-10-13.json`](experiments/metamorphic-crossbackend-hx370-v100-2026-10-13.json)
- [`metamorphic-crossdevice-v100-p40-2026-10-13.json`](experiments/metamorphic-crossdevice-v100-p40-2026-10-13.json)
- [`epyc-cuda13-compatibility.json`](experiments/epyc-cuda13-compatibility.json)
- [`tpu-vs-cpu-comparison.json`](../evidence/tpu/tpu-vs-cpu-comparison.json)
- [`tpu-jit-evidence.json`](../evidence/tpu/tpu-jit-evidence.json)
- [`tpu-campaign.json`](../evidence/tpu/tpu-campaign.json)
- [`manifest.json`](../evidence/tpu/manifest.json)
- [`evidence/tpu-v4/manifest.json`](../evidence/tpu-v4/manifest.json)
- [`evidence/tpu-v5p/manifest.json`](../evidence/tpu-v5p/manifest.json)
- [`evidence/tpu-v6e/manifest.json`](../evidence/tpu-v6e/manifest.json)

---

## v3.1.0 Strategic Parity, Snapshot, and Multi-Engine Ecosystem Roadmap

| Priority | Capability | Component | Status | Description |
|---|---|---|---|---|
| **116** | **ULP Distance & Statistics** | `xlaxiom.ulp` | **Implemented** | Bit-level integer ULP distance across float16, float32, and float64 with subnormal and signed zero equivalence (`ulp_distance`, `max_ulp_error`, `assert_ulp_close`). |
| **117** | **Golden Snapshot Testing** | `xlaxiom.golden` & `pytest_plugin` | **Implemented** | Record-once, compare-forever golden file snapshots with SHA-256 tamper-evident metadata and `--xlaxiom-update-golden` pytest fixture (`check_golden`, `save_golden`, `load_golden`). |
| **118** | **Dependency-Free Safetensors & BF16 IO** | `xlaxiom.safetensors_io` | **Implemented** | Pure-NumPy zero-dependency `.safetensors` reader and writer with exact BFloat16 bit-widening and automatic sharded index resolution (`model.safetensors.index.json`). |
| **119** | **Turnkey GitHub Action** | `action.yml` | **Implemented** | Composite GitHub Action (`Hardonian/XLAxiom@v3`) for one-line numerical regression gating, step summary generation, and sticky PR commenting bot. |
| **120** | **Top-K Logprob Parity for Serving** | `xlaxiom.serving` | **Implemented** | Distributional top-k next-token logprob auditing across OpenAI-compatible serving endpoints (vLLM, SGLang, HuggingFace causal models) with tokenizer normalization (`audit_topk_parity`). |
| **121** | **Multi-Tensor & JUnit XML CI Gate** | `xlaxiom.cli` (`ci-gate`, `snapshot`) | **Implemented** | Multi-tensor `.npz` and directory evaluations for CI gate with JUnit XML output generation for GitHub Actions and GitLab CI test UIs. |
| **122** | **Full Multi-Platform CI Matrix** | `.github/workflows/ci.yml` | **Implemented** | Automated Windows CI runner (`windows-latest`), PyTorch FX integration testing, and packaging smoke verification across operating systems. |

