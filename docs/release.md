# Release and schema policy

## Release criteria

A release requires clean formatting, linting, typing, tests, an sdist/wheel build, and
the CPU capability/property checks in CI. Any stated CUDA or TPU support additionally
requires target-hardware evidence; a successful CPU run is never substitute evidence.

Release publishing requires maintainer credentials. The repository can produce and
verify an OpenSSL detached evidence signature with `xlaxiom attest sign` and
`xlaxiom attest verify`; private-key handling remains external and no unsigned artifact
is represented as publisher-signed.

## Schema compatibility

Case, relational-case, bundle, corpus, metamorphic evidence/comparison, environment,
and report formats carry explicit versions.
Readers reject unknown versions instead of treating a changed artifact as an older one.
Compatible additions retain the current version; incompatible changes introduce a new
schema and a tested one-way migration command before the writer default changes. The
v1 JSON-schema documents are in [`schemas/`](schemas/).
