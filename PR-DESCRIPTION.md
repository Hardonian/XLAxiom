# XLAxiom 0.5.1 package hardening

## Scope

- Preserves the completed fifty-priority product and its retained hardware evidence.
- Marks the wheel as a PEP 561 typed distribution and modernizes license metadata.
- Adds explicit CUDA 12/13 install extras and complete installation guidance.
- Adds dependency maintenance and guarded, tokenless PyPI publication automation.

## Verification

- Formatting, lint, typing, complete tests, lockfile consistency, and distribution builds pass.
- Twine validates both distributions; the runtime dependency audit reports no known vulnerabilities.
- The wheel is installed and exercised under Python 3.11, 3.12, and 3.13.

## Evidence boundary

Physical TPU execution and macOS execution remain hardware/platform-unverified. PyPI publication still requires a maintainer to configure the protected GitHub environment and PyPI pending publisher, then deliberately publish a GitHub Release.
