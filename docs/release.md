# Release and schema policy

## Release criteria

A release requires clean formatting, linting, typing, tests, an sdist/wheel build, and
the CPU capability/property checks in CI. Any stated CUDA or TPU support additionally
requires target-hardware evidence; a successful CPU run is never substitute evidence.

Release publishing requires maintainer credentials. The repository can produce and
verify an OpenSSL detached evidence signature with `xlaxiom attest sign` and
`xlaxiom attest verify`; private-key handling remains external and no unsigned artifact
is represented as publisher-signed.

The `Release verification` workflow runs on `v*` tags and manual dispatch. It repeats
lint, format, typing, and test gates, verifies that a tag equals the packaged version,
installs the wheel into a fresh environment, exercises package data through the CLI,
and retains both wheel and source distribution as workflow artifacts. It deliberately
does not publish to PyPI or create a GitHub release without maintainer credentials.

Version 0.5.0 met the source, wheel, CPU, Windows/Linux, and available CUDA hardware
criteria on 2026-10-13. Physical TPU and macOS execution remain unverified support
boundaries, not release blockers or implied claims; see [`support.md`](support.md).

## Schema compatibility

Case, relational-case, bundle, corpus, metamorphic evidence/comparison, environment,
and report formats carry explicit versions.
Readers reject unknown versions instead of treating a changed artifact as an older one.
Compatible additions retain the current version; incompatible changes introduce a new
schema and a tested one-way migration command before the writer default changes. The
v1 JSON-schema documents are in [`schemas/`](schemas/).
