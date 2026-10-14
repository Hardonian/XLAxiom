# XLAxiom 0.6.0 release milestone

## Scope

- Lands the shared execution plan across NumPy and JAX eager/JIT execution with the validated NumPy-compatible adapter.
- Hardens integer comparisons at signed extrema against overflow and false passes under small tolerances.
- Enforces intermediate memory limits accounting for NumPy dtype promotion in mixed-dtype operations.
- Bundles the JAX #40634 historical record via package resources for portable `corpus build` execution from wheels and source checkouts.
- Removes dead code across hardware, corpus, gpu_validation, and historical modules.
- Hardens version matrix execution against argument injection in JAX specifiers.
- Optimizes CUDA variant selection and operation resolution hot paths.
- Pushes test coverage across the spawned worker boundary and relational artifact pipeline to 100% of classified paths and failure modes.
- Corrects relative-error metric scaling against element reference magnitudes and validates the benchmark measurement budget.

## Verification

- Formatting, lint, typing, complete tests (330 passed, 2 hardware-gated skipped), lockfile consistency, and distribution builds pass cleanly.
- Twine validates both source and wheel distributions; package smoke test passes end-to-end.

## Evidence boundary

Physical TPU execution and macOS execution remain hardware/platform-unverified. PyPI publication still requires a maintainer to configure the protected GitHub environment and PyPI pending publisher, then deliberately publish a GitHub Release.
