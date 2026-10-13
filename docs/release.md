# Release and schema policy

## Release criteria

A release requires clean formatting, linting, typing, tests, an sdist/wheel build, and
the CPU capability/property checks in CI. Any stated CUDA or TPU support additionally
requires target-hardware evidence; a successful CPU run is never substitute evidence.

Release publishing, signing, and registry attestation require maintainer credentials.
Until those are supplied, artifacts are integrity-checked with SHA-256 but are not
represented as publisher-signed releases.

## Schema compatibility

Case, relational-case, bundle, corpus, and report formats carry explicit versions.
Readers reject unknown versions instead of treating a changed artifact as an older one.
Compatible additions retain the current version; incompatible changes introduce a new
schema and a tested one-way migration command before the writer default changes. The
v1 JSON-schema documents are in [`schemas/`](schemas/).
