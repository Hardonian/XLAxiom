# Evidence artifacts and engineering handoff

XLAxiom artifacts are designed to make a numerical observation reviewable, portable, and retestable. They preserve the distinction between an observed discrepancy and a confirmed compiler defect.

## Recommended handoff package

For a real observation, share these together:

| Artifact | Create with | Why it matters |
| --- | --- | --- |
| Minimized case JSON | `xlaxiom minimize` | Records the smallest known predicate and its deterministic workload identity. |
| Reproduction verdict | `xlaxiom reproduce` | Shows whether the saved predicate still holds in a fresh command invocation. |
| JSON and Markdown report | `xlaxiom report` | Captures the comparison, numerical policy, execution outputs, and a readable summary. |
| Validated bundle | `xlaxiom bundle create` then `bundle validate` | Packages one or more checksummed cases for safe, portable review. |
| Version matrix, when relevant | `xlaxiom matrix run` | Separates a likely version transition from a stable observation. |

Never use a synthetic control as evidence of a JAX, XLA, hardware, or model defect. Synthetic controls are valuable only for testing the pipeline itself.

## Minimal review workflow

```console
# Validate the package before execution.
uv run xlaxiom bundle validate --bundle handoff.zip

# Re-run the archived cases without extracting them.
uv run xlaxiom bundle run --bundle handoff.zip --output results/handoff-evidence.json
```

Bundle validation rejects unsafe archive paths, duplicate or undeclared members, oversized uncompressed content, checksum mismatches, and malformed case schemas. Version 1 bundles contain only their manifest and declared case files. Bundle archives are deterministic for identical ordered inputs, which makes them suitable for content-addressed retention and review. Running a bundle produces a new evidence record; it does not mutate the archive.

Generated case, JSON evidence, and Markdown report files are written atomically: a complete temporary file is flushed beside its destination and then replaced in one operation. An interrupted write therefore leaves the prior artifact intact rather than a misleading partial record.

## Interpretation rules

1. **Start with provenance.** Confirm that the case is non-synthetic and identify the comparison mode and expected classification.
2. **Read the numerical policy.** Tolerances, dtype, non-finite behavior, and signed-zero handling are part of the evidence, not incidental settings.
3. **Treat environment as data.** Compare JAX/jaxlib, Python, OS, device, and relevant runtime configuration before inferring a regression.
4. **Use classifications conservatively.** A mismatch or version transition directs investigation; it does not establish cause, severity, or correctness.
5. **Protect machine details.** Remove credentials, private paths, proprietary inputs, and other sensitive data before external sharing. See [SECURITY.md](../SECURITY.md).

## Suggested issue or review summary

```text
Observation: <what changed and between which modes/environments>
Reproduction: <fresh-process verdict and exact command>
Artifact: <case JSON, report, and/or bundle with checksum>
Environment: <XLAxiom, JAX/jaxlib, Python, OS, device, relevant settings>
Scope: <what this evidence establishes and what it does not>
```

The repository’s [issue form](../.github/ISSUE_TEMPLATE/bug_report.yml) follows this structure so that maintainers and upstream teams receive actionable evidence from the start.
