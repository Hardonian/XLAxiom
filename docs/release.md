# Release and schema policy

## Release state

The source tree is prepared for version 0.6.1. It becomes a release only when the
annotated `v0.6.1` tag passes the `Release verification` workflow and its GitHub Release
is published. Do not describe a source-only or merely tagged version as published until
the PyPI publishing job has succeeded. Citation metadata is versioned with the release
candidate so the tagged source, wheel, source distribution, and citation agree.

## Release criteria

A release requires clean formatting, linting, typing, tests, an sdist/wheel build, and
the CPU capability/property checks in CI. Any stated CUDA or TPU support additionally
requires target-hardware evidence; a successful CPU run is never substitute evidence.

Release publishing requires maintainer credentials. The repository can produce and
verify an OpenSSL detached evidence signature with `xlaxiom attest sign` and
`xlaxiom attest verify`; private-key handling remains external and no unsigned artifact
is represented as publisher-signed.

The `Release verification` workflow runs on `v*` tags, GitHub Release publication,
and manual dispatch. It repeats
lint, format, typing, and test gates, verifies that a tag equals the packaged version,
validates package metadata, installs the wheel under Python 3.11–3.13 and the source
distribution under Python 3.12, exercises package data through the CLI, verifies
repeat builds byte-for-byte,
and retains both wheel and source distribution as workflow artifacts. It deliberately
does not create a GitHub release.

Pull requests and pushes first verify the committed lockfile and use frozen dependency
installs. Always-run artifact uploads retain JUnit and JSON capability/property evidence
that was produced before a failed check. A separate CI package-smoke job builds the
distributions, validates their metadata, then installs the wheel and source distribution
into fresh environments outside the checkout and exercises the CLI end to end.

Both build passes receive `SOURCE_DATE_EPOCH` from the tagged commit timestamp. This
normalizes archive headers across runner operating systems instead of relying on wall
clock timing.

Before creating a tag, run the same distribution-identity check locally. It verifies the
wheel and source distribution metadata, package data, and (when supplied) tag against
the package version:

```console
uv build --out-dir .release-dist
uv run twine check .release-dist/*
uv run python scripts/verify_distribution.py --dist-dir .release-dist --tag v0.6.1
```

Publishing a GitHub Release activates a separate, minimal PyPI job only after the
verification job passes. That job downloads the verified distributions and uses OIDC
Trusted Publishing from the protected `pypi` environment; it stores no long-lived PyPI
token. Before the first publication, a maintainer must configure a PyPI pending
publisher for `Hardonian/XLAxiom`, workflow `release.yml`, environment `pypi`, and must
configure required reviewers on that GitHub environment.

Version 0.6.0 met the source, wheel, CPU, Windows/Linux, and available CUDA hardware
criteria on 2026-10-13. Physical TPU and macOS execution remain unverified support
boundaries, not release blockers or implied claims; see [`support.md`](support.md).

## Schema compatibility

Case, relational-case, bundle, corpus, metamorphic evidence/comparison, environment,
and report formats carry explicit versions.
Readers reject unknown versions instead of treating a changed artifact as an older one.
Compatible additions retain the current version; incompatible changes introduce a new
schema and a tested one-way migration command before the writer default changes. The
v1 JSON-schema documents are in [`schemas/`](schemas/).
