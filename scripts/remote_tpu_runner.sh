#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: System info & Python packages ==="
sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git curl

echo "=== Step 2: Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Step 3: Clone XLAxiom ==="
rm -rf XLAxiom
git clone https://github.com/Hardonian/XLAxiom.git
cd XLAxiom

echo "=== Step 4: Python 3.12 Virtualenv & JAX TPU Installation ==="
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Step 5: Silicon Self-Test ==="
python -c "import jax; print('JAX Version:', jax.__version__); print('JAX Devices:', jax.devices()); assert any(d.platform == 'tpu' for d in jax.devices()), 'NO TPU FOUND!'"

echo "=== Step 6: Evidence Generation ==="
mkdir -p evidence/tpu
python -m xlaxiom.cli doctor > evidence/tpu/doctor.json
python -m xlaxiom.cli environment capture --label "google-cloud-tpu-v5e-viperlite" --output evidence/tpu/environment.json

echo "Running 40-property metamorphic suite on TPU eager & JIT..."
python -m xlaxiom.cli property run --backend jax-tpu-eager --output evidence/tpu/tpu-eager-evidence.json
python -m xlaxiom.cli property run --backend jax-tpu-jit --output evidence/tpu/tpu-jit-evidence.json
python -m xlaxiom.cli property campaign --cases 50 --backend jax-tpu-jit --output evidence/tpu/tpu-campaign.json
python -m xlaxiom.cli property run --backend jax-jit --output evidence/tpu/cpu-jit-evidence.json
python -m xlaxiom.cli property compare --baseline evidence/tpu/cpu-jit-evidence.json --comparison evidence/tpu/tpu-jit-evidence.json --output evidence/tpu/tpu-vs-cpu-comparison.json

echo "Running Higher-Order Autodiff on TPU..."
python -m xlaxiom.cli gradient --higher-order --seed 0 --output evidence/tpu/tpu-higher-order.json || true

echo "=== TPU EVIDENCE RUN COMPLETED SUCCESSFULLY ==="

