# Research landscape and scope

Research performed 2026-10-12 from official documentation and upstream project materials. This document intentionally avoids claims about unverified versions, defects, or benchmarks.

| Project / source | Existing capability | XLAxiom contribution |
| --- | --- | --- |
| [JAX](https://docs.jax.dev/en/latest/quickstart.html) | NumPy-like computation, transformations, and JIT compilation through OpenXLA. | Records bounded, declarative programs and compares eager/JIT and NumPy observations as reproducible evidence. |
| [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) | Documents CPU and platform-dependent accelerator packages. | Capability-gated backend reporting: availability is measured on the host, not inferred from a machine name. |
| [JAX x64 documentation](https://docs.jax.dev/en/latest/101/default_dtypes.html) | Explains that x64 is disabled by default and controlled by configuration. | Treats dtype and precision configuration as evidence fields and comparison-policy inputs. |
| [NumPy comparison documentation](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html) | Defines tolerance-based array comparison and NaN handling. | Adds explicit classified outcomes, shape/dtype handling, signed-zero checks, and reportable metrics. |
| [Hypothesis](https://hypothesis.readthedocs.io/) | General property-based testing and NumPy strategies. | Uses a persisted domain-specific workload language, backend execution metadata, and failure-preserving reducer rather than test-framework-specific examples. |
| [C-Reduce](https://embed.cs.utah.edu/creduce/) | Delta debugging and reduction of compiler test cases. | Applies bounded, validity-aware reduction to numerical operation graphs with a numerical failure predicate. |

## Literature and design conclusions

Differential and metamorphic testing are established approaches for compiler/numerical investigation, but raw divergence is rarely enough for upstream triage. Floating-point reordering, dtype defaults, optimized kernels, and unsupported modes can all create expected variation. XLAxiom’s initial value is therefore the evidence chain: a safe portable workload, explicit policy, observed environment, predicate-preserving reduction, and independent reproduction command.

The initial implementation does not claim a new fuzzing algorithm, an independent oracle for all JAX semantics, or a real compiler finding. Near-term research value lies in adding metamorphic relations, collecting independently reproduced reduced cases, and validating device/version adapters in isolated environments.
