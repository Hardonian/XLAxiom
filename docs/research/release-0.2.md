# XLAxiom 0.2 research validation report

## Method

The 0.2 campaign used deterministic seeds 1000–1499 with four bounded graph motifs: elementwise composition, broadcasting plus reduction, matrix multiplication plus transform/reduction, and reshape/unary/reshape. Each workload underwent symbolic validation before allocation. The campaign compared NumPy eager output against explicitly device-placed JAX CPU JIT output, then JAX CPU eager against JIT. All use the case’s `dtype-aware-v1` policy.

Generated trusted workloads ran in-process for throughput. Persisted or externally supplied cases can use `xlaxiom.safety.compare_isolated`, which starts a Windows/Linux spawned child and returns structured timeout/crash records; no arbitrary executable payload is read from a case.

## Actual results

The record [`../experiments/cpu-campaign-2026-10-12.json`](../experiments/cpu-campaign-2026-10-12.json) contains all counts and parameters.

- 500 requested / 500 completed CPU workloads.
- 500 NumPy-vs-JIT passes; zero discrepancies, exceptions, unsupported backends, or inconclusive observations.
- 500 eager-vs-JIT passes; zero discrepancies, exceptions, unsupported backends, or inconclusive observations.
- Duration: 39.7335 seconds on Windows 11, JAX 0.11.2/jaxlib 0.11.2 CPU.
- No candidate anomalies were retained or minimized because none were observed.

The historical adapter provides separate evidence: JAX 0.11.1 returned zero for both large-float32 gradients from issue #40634, while the local JAX 0.11.2 still returned zero for `arcsinh` and a correct nonzero value for `acosh`. See [the dedicated report](jax-40634.md).

## Safety and interpretation

The graph validator rejects rank overflow, individual and cumulative allocation budgets, incompatible broadcasting, invalid matrix contraction, and unsafe shape products before materialization. JAX input arrays are placed onto the requested CPU device; output device placement is checked and reported. A comparison observation is not a root-cause assertion.

The 500-case corpus is intentionally small and CPU-only. It does not establish accelerator behavior, prove the absence of compiler defects, or replace isolated environment/version validation. The historical finding is already tracked upstream, and the known upstream fix must be matched to release tags before users infer a release-status claim.

## Upstream readiness

No upstream issue or pull request was filed. The useful human-reviewed proposal is to verify which post-0.11.2 release first contains [commit 86d3fe7](https://github.com/jax-ml/jax/commit/86d3fe7), then ensure a release-level regression test covers both `arcsinh` and `acosh` at `float32(1e22)`.
