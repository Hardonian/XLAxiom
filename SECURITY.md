# Security policy

## Report privately

Do not open a public issue for a vulnerability that could expose credentials, disclose
private machine details, execute unintended code, bypass artifact validation, or cause
uncontrolled resource consumption. Use the repository's [private vulnerability reporting
flow](https://github.com/Hardonian/XLAxiom/security/advisories/new) and include:

- the affected XLAxiom version or commit;
- a minimal, non-sensitive reproduction and its expected impact;
- the operating system, Python, JAX/jaxlib, and accelerator context when relevant; and
- any safe mitigation you have already identified.

Never include credentials, private paths, proprietary model inputs, raw GPU UUIDs, or
other secrets in a report. If GitHub private reporting is unavailable, contact the
repository maintainer privately through the GitHub profile rather than creating a public
issue.

## Supported security surface

Security fixes are made against the latest tagged release and the current `main` branch.
Older releases may receive guidance, but should be upgraded before relying on a fix.

The most relevant report classes are:

- unsafe parsing, archive traversal, checksum bypass, or arbitrary-code execution;
- unexpected disclosure in generated reports, traces, environment records, or bundles;
- a capability gate that silently falls back to a different backend or device; and
- resource exhaustion or cleanup failures caused by a schema-valid artifact.

## Operational boundary

XLAxiom accepts declarative JSON only: workloads use a closed operation registry and it
does not deserialize pickle files or execute workload-supplied Python. Bundles are
validated before use and generated artifacts are checksummed.

Those controls are not a substitute for sandboxing. Numerical compilation and execution
can consume CPU, memory, disk, or accelerator resources, particularly for unfamiliar
artifacts. Run untrusted cases in an appropriately isolated process and environment;
use bounded/isolated campaign modes where available, and do not expose production
credentials or privileged devices to an untrusted case.

## Disclosure

Maintainers will validate a report privately, coordinate a fix and release when needed,
and then disclose the issue with appropriate credit unless the reporter requests
otherwise. Please allow a coordinated remediation path before public discussion.

