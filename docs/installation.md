# Installation

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

## Accelerator extras

After PyPI publication, Linux users may request the matching JAX CUDA family with
`xlaxiom[cuda12]` or `xlaxiom[cuda13]`. These extras install user-space JAX packages;
they do not install an NVIDIA driver or change system CUDA. Always begin with:

```console
xlaxiom gpu doctor
xlaxiom gpu validate
```

CUDA compatibility remains capability-gated, and an unavailable or incompatible GPU
is reported instead of silently falling back to CPU. TPU execution is implemented but
has no installation extra or physical verification claim yet.
