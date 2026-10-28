# RFC: Automated OpenXLA StableHLO Compiler Pass Bisection and Fault Isolation

* **Status**: Proposed
* **Author**: XLAxiom Core Team
* **Target**: OpenXLA Architecture & Compiler Working Group

## Abstract
Modern optimizing compilers for accelerated hardware (such as OpenXLA / XLA) execute multi-stage transformation pipelines: parsing high-level StableHLO dialects, running optimization passes (e.g. `algebraic-simplifier`, `instruction-fusion`, `layout-assignment`, `rematerialization`), and generating hardware machine code.

When numerical discrepancies or compiler crashes occur, identifying which specific pass mutated or corrupted semantics typically requires manual flag manipulation (`--xla_disable_hlo_passes=...`).

We propose an automated pass bisection protocol:
1. **Pass Sequence Model**: Represent XLA transformation passes as an ordered, bisection-amenable pipeline.
2. **Intermediate Parity Checking**: Compare program outputs after each pass against the unoptimized StableHLO reference interpreter (`xlaxiom.stablehlo_interpreter`).
3. **Automated Blame Assignment**: Pinpoint the precise pass and intermediate HLO instruction where numerical drift exceeds designated tolerances.

## Workflow Integration
```bash
# Pinpoint divergent XLA optimization pass
xlaxiom xla pass-isolate --case divergent_case.json --output pass_isolation.json
```

## Community Benefits
- Reduces compiler bug triage from days to minutes.
- Guarantees actionable reproduction packages with minimal SSA def-use chains.
- Standardizes compiler test coverage across OpenXLA backends.

