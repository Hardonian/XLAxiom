# Pull Request: Integrate XLAxiom Standalone Repro Scripts into Issue Templates

## Overview
This PR enhances the JAX bug report issue template (`.github/ISSUE_TEMPLATE/bug_report.yml`) to support **XLAxiom automated standalone reproduction packages**.

## Problem
A significant fraction of JAX compiler triage time is spent converting user bug reports into minimal, self-contained repro scripts without external dependencies, capturing exact JAX environment metadata (`jax.print_environment_info()`), and isolating whether a bug occurs in eager mode, JIT, or lower-level StableHLO lowering.

## Proposed Solution
Provide an optional but first-class field in the JAX issue template for `XLAxiom Repro Scripts`. Users can run:

```bash
xlaxiom triage --case issue_case.json --target jax
```

This generates:
1. A **10-line standalone reproduction script** with no external dependencies beyond `jax` and `numpy`.
2. Exact environment provenance and hardware ordinals via `jax.print_environment_info()`.
3. Standalone OpenXLA StableHLO MLIR text representation.
4. A 1-click **Open In Colab** badge reproducing the discrepancy with hardware acceleration.

## Changes Included
- Added `xlaxiom_repro` textarea to `.github/ISSUE_TEMPLATE/bug_report.yml`.
- Added automated CI verification action for validating reported repro scripts.

