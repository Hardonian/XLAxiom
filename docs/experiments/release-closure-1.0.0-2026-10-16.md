# XLAxiom 1.0.0 General Availability release closure audit — 2026-10-16

## Outcome

XLAxiom 1.0.0 General Availability (GA) is complete across all strategic priorities,
compiler integrations, and distribution packages. Every capability planned across
the 61 holistic delivery milestones is implemented, verified, and audited against
strict quality gates.

## Acceptance evidence

| Verification Surface | Gate / Tool | Result |
| --- | --- | --- |
| Full test suite | `pytest` | 378 passed, 0 failed, 2 expected capability-gated skips across unit, integration, and E2E suites |
| Code quality & linter | `ruff check .` | 0 violations across the entire repository |
| Code formatting | `ruff format --check .` | 120 files verified clean |
| Static typing | `mypy src` | Success: no issues found in 42 source files under strict typing |
| Package wheel | `uv build` | Built `xlaxiom-1.0.0-py3-none-any.whl`, verified package data, schemas, and entry points |
| Source distribution | `uv build` | Built `xlaxiom-1.0.0.tar.gz`, verified PKG-INFO, metadata, and license data |
| Distribution metadata | `twine check` | Wheel and sdist PASSED metadata audit |
| Distribution integrity | `verify_distribution.py` | Verified `xlaxiom 1.0.0` wheel, sdist, and release tag `v1.0.0` |
| Curated corpus | `property run / sharpness` | 35 declarative metamorphic relations verified with 60%–100% mutation sharpness |
| Tracing & Ingestion | `trace_jax`, `extract_subgraph` | Live JAX callable ingestion, topological dependency closure extraction, and `@capture` decorator |
| Compiler Root-Cause | `xlaxiom bisect` | Automated binary search across isolated `uv` virtual environments |
| Upstream Triage | `xlaxiom triage` | Standalone 10-line reproduction script generation and markdown issue packaging |
| Multi-Compiler | PyTorch Inductor | Eager and `torch.compile` differential execution with device placement records |
| Hardware discovery | Apple Silicon & AMD ROCm | Non-destructive hardware discovery with strict fail-closed enforcement (no silent CPU fallback) |
| Developer ergonomics | `pytest-xlaxiom` | Native plugin registered via `pytest11` entry-point with markers, fixtures, and HTML reports |
| CI automation | GitHub Action | Turnkey composite action `.github/actions/xlaxiom-action/action.yml` with step summaries |
| IDE diagnostics | IDE Problems API | Clean closure: 0 errors, 0 warnings across all source and documentation files |

## Metamorphic Corpus Coverage & Sharpness

The curated corpus expanded from 24 to 35 declarative invariants, covering algebraic, lattice,
and floating-point edge-case semantics:

- Algebraic commutativity, associativity, and distributivity across floating-point and integer types.
- Lattice absorption identities ($\max(x, \min(x, y)) = x$) and idempotence properties.
- Dualities for minimum/maximum, negation involution, and global mean/sum reductions.
- Signed-zero strict invariance (`check_signed_zero: true`) without numerical drift.
- Mutation sharpness confirmed across all properties, maintaining 60%–100% mutant kill rates
  against synthetic compiler mutation hypotheses.

## Publication boundary

The source tree, packaging configurations, and distribution artifacts are synchronized for version `1.0.0`:
- `pyproject.toml` version `1.0.0`
- `src/xlaxiom/_version.py` version `1.0.0`
- `CITATION.cff` version `1.0.0`, release date `2026-10-16`
- `CHANGELOG.md` entry for `1.0.0`
- `docs/roadmap.md` complete status across all 61 priorities

Publishing to PyPI activates via GitHub Actions using tokenless OIDC Trusted Publishing
once maintainers create and push the annotated `v1.0.0` tag.
