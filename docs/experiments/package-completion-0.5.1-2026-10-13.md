# XLAxiom 0.5.1 package completion audit — 2026-10-13

## Outcome

The package layer is complete for source and verified-artifact distribution. This
follow-up does not change the numerical evidence claims in 0.5.0.

## Acceptance evidence

| Scope | Result |
| --- | --- |
| Python source tests | 257 passed and 2 expected no-GPU skips on each of Python 3.11, 3.12, and 3.13 |
| Static gates | Ruff lint/format and mypy passed |
| Distribution metadata | Wheel and sdist passed Twine validation |
| Installed wheel | Version, property registry, corpus validation, and package resources passed on Python 3.11–3.13 outside the checkout |
| Runtime dependencies | No known vulnerabilities reported by `pip-audit` at audit time |
| Reproducibility | Two wheel builds and two sdist builds were byte-for-byte identical |
| Workflow syntax | All GitHub Actions workflows passed `actionlint` |
| Documentation | All local Markdown link targets resolved |

The wheel contains SPDX `License-Expression: Apache-2.0`, the license file, the
`Typing :: Typed` classifier, the PEP 561 `py.typed` marker, the curated property
corpus, and explicit Linux `cuda12` and `cuda13` extras.

The first `v0.5.1` tag run exposed wall-clock gzip header variance on the Linux runner,
despite repeat builds matching locally. Version 0.5.2 corrects the release workflow by
setting `SOURCE_DATE_EPOCH` from the tagged commit. Two independent EPYC Linux builds
then produced identical wheel and sdist bytes before the corrective tag was created.
The `v0.5.2` run then showed that an in-checkout repeat output directory itself alters
sdist selection on Linux. Version 0.5.3 moves that output to `RUNNER_TEMP`, matching the
already-successful independent EPYC procedure and keeping both build destinations
outside one another's source context.

## Publication boundary

PyPI did not contain an `xlaxiom` project at audit time. The release workflow is ready
for tokenless OIDC Trusted Publishing and automatic PyPI attestations, but it cannot
publish until a maintainer configures the `pypi` GitHub environment and PyPI pending
publisher, then deliberately publishes a GitHub Release. No package-index publication
is claimed by this audit.
