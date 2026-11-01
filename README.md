# XLAxiom — Universal Numerical Verification & Model Diffing

[![PyPI version](https://img.shields.io/pypi/v/xlaxiom.svg?color=blue)](https://pypi.org/project/xlaxiom/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/xlaxiom.svg?color=brightgreen)](https://pypi.org/project/xlaxiom/)
[![Python 3.11–3.13](https://img.shields.io/pypi/pyversions/xlaxiom.svg)](https://pypi.org/project/xlaxiom/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml/badge.svg)](https://github.com/Hardonian/XLAxiom/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**XLAxiom is the standard numerical verification, model diffing, and differential testing toolkit for modern AI systems.** Built on a zero-dependency pure-NumPy core (`numpy>=1.24`), it provides bit-exact ULP error analysis, zero-code `pytest` assertion interception, Safetensors model weight diffing, checkpoint sanity linting, and FP8 quantization risk profiling.

It gives AI engineers, compiler developers, and ML infrastructure teams a disciplined path from "this changed" to an evidence-backed diagnostic artifact another engineer can inspect, rerun, and compare.

It is independently developed and is not affiliated with, sponsored by, or endorsed by Google.

---

## Why XLAxiom?

| Feature | `np.testing.assert_allclose` | `torch.testing.assert_close` | **XLAxiom** |
| :--- | :---: | :---: | :---: |
| **Signal-to-Noise Ratio (SNR in dB)** | ❌ | ❌ | ✅ **Exact in dB** |
| **Cosine Similarity** | ❌ | ❌ | ✅ **Exact [-1.0, 1.0]** |
| **Bit-exact ULP Distance** | ❌ | ❌ | ✅ **Float16, BF16, Float32, Float64** |
| **Subnormal / Denormal Detection** | ❌ | ❌ | ✅ **Detects 10x-100x latency traps** |
| **Catastrophic Cancellation Flag** | ❌ | ❌ | ✅ **Automated root-cause diagnosis** |
| **Error Distribution Percentiles** | ❌ | ❌ | ✅ **p50, p90, p99, RMS, Max** |
| **Minimal Reproduction Snippet** | ❌ | ❌ | ✅ **Generated copy-pasteable script** |
| **Offline Interactive HTML Reports** | ❌ | ❌ | ✅ **Standalone SVG & CSS dashboard** |
| **Zero-Code Pytest Interception** | ❌ | ❌ | ✅ **`pytest --xlaxiom-rich-asserts`** |
| **Safetensors Checkpoint Diffing** | ❌ | ❌ | ✅ **`xlaxiom diff a.safetensors b.safetensors`** |
| **Checkpoint Health & NaN Linter** | ❌ | ❌ | ✅ **`xlaxiom lint-checkpoint`** |
| **FP8 / INT4 Quantization Profiler**| ❌ | ❌ | ✅ **`xlaxiom quant-audit`** |
| **Turnkey Pre-commit & CI Action** | ❌ | ❌ | ✅ **`.pre-commit-hooks.yaml` & `action.yml`** |
| **Zero Heavy Dependencies** | ✅ | ❌ (requires torch) | ✅ **Pure NumPy core (`numpy>=1.24`)** |

---

## ⚡ Supercharged 1-Minute Workflows

### 1. Upgrade Any Existing Test Suite with Zero Code Changes

Supercharge every `torch.testing.assert_close` and `np.testing.assert_allclose` across your repository to output rich SNR, cosine similarity, bitwise ULP, and root-cause diagnostics:

```bash
pytest --xlaxiom-rich-asserts --xlaxiom-save-html
```

Or enable it globally in your root `conftest.py`:

```python
import xlaxiom as xl

xl.enable_rich_asserts(save_html=True)
```

### 2. Diff Any Two Model Checkpoints or Weights Files

Inspect layer-by-layer parameter drift between base and fine-tuned models, merged models, or quantized weights:

```bash
# Compare two Safetensors models with top-10 divergence table and interactive HTML dashboard
xlaxiom diff model_v1.safetensors model_v2.safetensors --top-k 10 --html diff.html
```

### 3. Checkpoint Health & Latency-Risk Linter

Detect silent model collapse (NaNs/Infs), dead weights, and subnormal floating-point numbers (which cause 10x-100x GPU/CPU inference latency spikes due to microcode assist traps):

```bash
# Pre-deployment or pre-commit checkpoint audit
xlaxiom lint-checkpoint model.safetensors --fail-on-warning
```

### 4. FP8 / INT4 Quantization Risk Profiling

Identify sensitive layers that will suffer precision collapse before quantizing your LLM:

```bash
# Audits entire checkpoint and outputs recommended BF16 fallback whitelist
xlaxiom quant-audit model.safetensors --dtype fp8_e4m3
```

---

## Everyday Python API

XLAxiom provides drop-in numerical assertions, custom kernel fuzzing, and layer parity auditing for daily ML development:

```python
import xlaxiom as xl

# 1. Drop-in assertion with root-cause diagnostics, SNR & cosine similarity
xl.assert_close(actual_tensor, reference_tensor, rtol=1e-4, atol=1e-5)

# 2. Bit-level ULP distance assertions across float16/bfloat16/float32/float64
xl.assert_ulp_close(actual_tensor, reference_tensor, max_ulp=2.0)

# 3. Record-once, verify-forever golden snapshot testing
xl.check_golden("model_golden.npz", output_tensor, max_ulp=4.0)

# 4. Checkpoint weight parity audit (Safetensors / .npz / PyTorch state_dict)
ckpt_report = xl.audit_checkpoint_parity("base.safetensors", "quantized.safetensors")
print(ckpt_report.format_summary())

# 5. Checkpoint health linter (NaNs, Infs, subnormals, dead weights)
health = xl.lint_checkpoint("model.safetensors")
print(health.format_summary())

# 6. Audit quantization precision loss and outlier clipping risk across model
quant_report = xl.audit_checkpoint_quantization("model.safetensors", target_dtype="fp8_e4m3")
print(quant_report.format_summary())

# 7. Differentially fuzz custom Triton, Pallas, or CUDA kernels across dynamic shapes
res = xl.fuzz_kernel(
    my_custom_kernel,
    reference_fn,
    shape_spec=((1, 8), (16, 128), [32, 64]),
    dtypes=["float32"],
)

# 8. Audit layer-by-layer activation parity between PyTorch, JAX, or compiled models
report = xl.audit_layer_parity(model_eager, model_compiled, sample_input)
print(report.format_table())

# 9. LLM serving engine parity verification (vLLM / SGLang / TensorRT-LLM vs Reference)
serving_report = xl.audit_serving_parity(vllm_engine, hf_reference, sample_prompts)
print(serving_report.format_summary())

# 10. Continuous Integration regression gate & PR comment generator
diff_obj = xl.diff(candidate_tensor, baseline_tensor)
gate = xl.evaluate_ci_gate([("model_output", diff_obj)], min_snr_db=30.0)
print(gate.summary_markdown)
```

---

## Pre-Commit Hook Integration

Add XLAxiom to your `.pre-commit-config.yaml` to ensure no corrupted checkpoints or regressions are committed:

```yaml
repos:
  - repo: https://github.com/Hardonian/XLAxiom
    rev: v3.1.0
    hooks:
      - id: xlaxiom-lint-checkpoint
      - id: xlaxiom-snapshot-check
```

---

## Turnkey GitHub Actions CI/CD Integration

Drop automated numerical regression gates into your GitHub workflows with zero boilerplate:

```yaml
- name: Verify Numerical Parity Gate
  uses: Hardonian/XLAxiom@v3
  with:
    actual: "candidate_weights.safetensors"
    desired: "reference_weights.safetensors"
    rtol: "1e-4"
    atol: "1e-5"
    min-snr: "25.0"
    comment-on-pr: "true"
```
    min-snr: "25.0"
    comment-on-pr: "true"
```

## Five-minute confidence check

Prerequisite: Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). From a source checkout:

```console
uv sync --extra dev --frozen
uv run xlaxiom doctor
uv run xlaxiom generate --synthetic --seed 7 --output results/original.json
uv run xlaxiom minimize --case results/original.json --output results/reduced.json
uv run xlaxiom reproduce --case results/reduced.json
uv run xlaxiom report --case results/reduced.json --output results/report
uv run xlaxiom bundle create --case results/reduced.json --output results/control.zip
uv run xlaxiom bundle validate --bundle results/control.zip
```

This produces a small, checksummed control case, a fresh-process reproduction verdict, a JSON/Markdown report, and a validated portable bundle. `--synthetic` injects a clearly labelled offset after JAX execution: it proves the workflow, but is never evidence of a JAX, XLA, hardware, or model defect.

For a real observation, start with the [artifact guide](docs/artifacts.md). It explains the required minimization, fresh-process reproduction, environment evidence, and safe handoff boundaries before an upstream issue is considered.

## What XLAxiom does

```text
observe → classify → minimize → reproduce in a fresh process
        → bundle evidence → compare environments
```

- **Records the evidence.** It captures versions, device placement, numerical policy, outputs, and an explicit predicate instead of treating a mismatch as a diagnosis.
- **Keeps cases small.** Failure-preserving minimization and topological subgraph extraction reduce a reproducible observation without silently changing what failed.
- **Pinpoints compiler regressions.** Automated bisection binary-searches compiler releases in isolated environments to identify exact breaking changes.
- **Makes handoffs portable.** Declarative JSON workloads, copy-pasteable 10-line reproduction scripts, triage markdown packages, checksummed reports, and bundles let another engineer or upstream maintainer validate the observation immediately.
- **Fails closed on hardware.** Accelerator requests must prove actual placement; unavailable GPU/TPU hardware is never represented as a CPU success.

XLAxiom is the evidence layer between a suspicious numerical result and a focused engineering investigation. It is not a general correctness oracle, performance benchmark suite, or automatic root-cause analyzer.

## The normal workflow

1. **Capture an observation.** Compare NumPy against JAX eager/JIT or execute a reviewed relation with an explicit policy.
2. **Reduce it without changing it.** Minimize a repeatable non-synthetic predicate and preserve its reduction record.
3. **Reproduce it independently.** Re-run in a fresh process, then record a second host, device, or runtime when the claim requires it.
4. **Hand it off safely.** Generate a report and validated bundle; attach only the evidence that supports the stated scope.

The [artifact lifecycle](docs/artifacts.md) has the exact command-level process. The [issue readiness check](docs/quickstart.md#before-escalating-a-real-observation) is deliberately conservative: an observation is not a diagnosis, and missing external evidence is never inferred.

## Evidence you can trust

The repository is designed to make scope visible rather than inflate it:

- The CPU path, checksummed artifacts, failure containment, and reviewed property corpus are covered by automated tests and release verification.
- The curated corpus contains forty explicit relations across ordinary, extreme, signed-zero, non-finite, integer, mixed-dtype, and x64 inputs with verified mutation sharpness.
- Windows/Linux, selected CUDA, and 4-chip Google Cloud TPU v5e evidence are retained as concrete records; untested hardware remains capability- and evidence-gated.

See the [support and verification matrix](docs/support.md) for what is supported versus merely implemented, and the [experiment records](docs/experiments/) for the underlying evidence—not a condensed lab notebook in this README.

## Command map

| Goal | Primary commands |
| --- | --- |
| Ingest live code | `trace` |
| Inspect an environment | `doctor`, `environment capture`, `runtime compare` |
| Investigate one case | `generate`, `run`, `compare`, `minimize`, `reproduce` |
| Pinpoint compiler regression | `bisect` |
| Prepare upstream triage/handoff | `triage`, `report`, `bundle create`, `bundle validate`, `readiness` |
| Test known relations | `property list`, `property run`, `property campaign`, `relation` |
| Compare changes safely | `matrix run`, `property compare`, `historical-study` |
| Validate accelerators | `gpu doctor`, `gpu validate`, `gpu campaign` |
| Google Ecosystem & Cloud TPUs | `cloud tpu-run`, `cloud tpu-status`, `cloud vertex-spec`, `cloud xpk-spec` |
| Developer surfaces & Colab | `colab`, `xla pass-isolate` |
| Enterprise Governance & Telemetry | `bigquery schema`, `bigquery export`, `kms sign`, `kms verify` |
| Universal Parity & Quantization | `diff`, `quant-audit` |

Run `xlaxiom <command> --help` for the complete interface. The [documentation index](docs/index.md) routes each workflow to the right level of detail.

## Safety and interpretation

Workloads are deterministic JSON data selected from a closed operation registry; XLAxiom does not load pickle files or execute workload-supplied code. Numerical workloads can still consume meaningful resources, so run untrusted artifacts with suitable process and resource isolation. Read [SECURITY.md](SECURITY.md) before sharing artifacts or reporting a vulnerability.

NumPy is useful but is not an independent oracle for every JAX semantic, dtype-promotion rule, compiler setting, or hardware implementation. A mismatch directs investigation; it never establishes root cause or severity by itself.

## Development and release

```console
uv sync --extra dev --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build --out-dir .release-dist
uv run twine check .release-dist/*
uv run python scripts/verify_distribution.py --dist-dir .release-dist
```

The [release policy](docs/release.md) defines the required quality gates, reproducible-build expectations, and Trusted Publishing boundary. The [roadmap](docs/roadmap.md) distinguishes implemented capability from physical-hardware evidence that cannot be manufactured locally.
