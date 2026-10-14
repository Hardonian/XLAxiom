# Installation

## From PyPI

After the stable release is published, install the exact release you intend to run:

```console
python -m pip install "xlaxiom==0.6.1"
xlaxiom --version
xlaxiom doctor
```

For an isolated command-line installation with `uv`:

```console
uv tool install "xlaxiom==0.6.1"
xlaxiom --version
```

Pinning the version makes saved evidence easier to reproduce. Upgrade deliberately after
reviewing the [changelog](../CHANGELOG.md).

## From a source checkout

Python 3.11, 3.12, and 3.13 are supported. A CPU development environment is the
default and does not modify system CUDA libraries:

```console
git clone https://github.com/Hardonian/XLAxiom.git
cd XLAxiom
uv sync --extra dev --frozen
uv run xlaxiom --version
uv run xlaxiom doctor
```

## From a verified wheel

Download the wheel from a successful `Release verification` workflow artifact, then
install that exact artifact into a fresh environment:

```console
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python path/to/xlaxiom-<version>-py3-none-any.whl
.venv/bin/xlaxiom --version
```

On Windows, use `.venv\\Scripts\\python.exe` and `.venv\\Scripts\\xlaxiom.exe`.
The wheel includes the curated and historical evidence records plus the `py.typed` marker.
Use `twine check` and the distribution verifier before installing a locally built artifact:

```console
uv build
uv run twine check dist/*
uv run python scripts/verify_distribution.py --dist-dir dist
```

## Accelerator extras

Linux users may request the matching JAX CUDA family with `xlaxiom[cuda12]` or
`xlaxiom[cuda13]`. These extras install user-space JAX packages;
they do not install an NVIDIA driver or change system CUDA. Always begin with:

```console
xlaxiom gpu doctor
xlaxiom gpu validate
```

CUDA compatibility remains capability-gated, and an unavailable or incompatible GPU
is reported instead of silently falling back to CPU. TPU execution is implemented but
has no installation extra or physical verification claim yet.
