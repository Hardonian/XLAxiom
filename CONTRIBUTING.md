# Contributing

Use Python 3.11–3.13 and `uv`. Keep a change focused and add tests that exercise real NumPy/JAX behavior where JAX is available.

Before opening a pull request, run:

```console
uv lock --check
uv sync --extra dev --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
uv run xlaxiom --version
```

Do not submit a synthetic-control observation as a compiler issue. A proposed upstream case needs a portable artifact, fresh-process reproduction, recorded device/version/configuration evidence, and a minimized predicate.
