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
rm -rf evidence/tpu
mkdir -p evidence/tpu
status_tsv=evidence/tpu/stage-status.tsv
: > "$status_tsv"
failed=0
run_stage() {
    stage="$1"
    shift
    echo "--- Running stage: $stage ---"
    set +e
    "$@" >"evidence/tpu/${stage}.log" 2>&1
    code=$?
    set -e
    printf '%s\t%s\n' "$stage" "$code" >> "$status_tsv"
    if [ "$code" -ne 0 ]; then
        failed=1
        echo "Stage '$stage' failed with exit code $code; retaining evidence and continuing."
        tail -n 20 "evidence/tpu/${stage}.log"
    else
        echo "Stage '$stage' passed."
    fi
}

run_stage doctor env PYTHONWARNINGS=ignore python -m xlaxiom.cli doctor
cp evidence/tpu/doctor.log evidence/tpu/doctor.json
run_stage environment python -m xlaxiom.cli environment capture --label "google-cloud-tpu-v5e-viperlite" --output evidence/tpu/environment.json
run_stage tpu-eager python -m xlaxiom.cli property run --x64 --backend jax-tpu-eager --output evidence/tpu/tpu-eager-evidence.json
run_stage tpu-jit python -m xlaxiom.cli property run --x64 --backend jax-tpu-jit --output evidence/tpu/tpu-jit-evidence.json
run_stage tpu-campaign python -m xlaxiom.cli property campaign --x64 --cases 50 --backend jax-tpu-jit --output evidence/tpu/tpu-campaign.json
run_stage cpu-jit env JAX_PLATFORMS=cpu python -m xlaxiom.cli property run --x64 --backend jax-jit --output evidence/tpu/cpu-jit-evidence.json
run_stage cpu-compare python -m xlaxiom.cli property compare --baseline evidence/tpu/cpu-jit-evidence.json --comparison evidence/tpu/tpu-jit-evidence.json --output evidence/tpu/tpu-vs-cpu-comparison.json
run_stage higher-order python -m xlaxiom.cli gradient --higher-order --seed 0 --output evidence/tpu/tpu-higher-order.json

STATUS_TSV="$status_tsv" python - <<'PY'
import json
import os
from pathlib import Path

rows = []
for line in Path(os.environ["STATUS_TSV"]).read_text().splitlines():
    stage, code = line.split("\t", 1)
    rows.append({"stage": stage, "exit_code": int(code), "passed": code == "0"})
Path("evidence/tpu/stage-status.json").write_text(
    json.dumps({"kind": "physical-tpu-stage-status", "passed": all(row["passed"] for row in rows), "stages": rows}, indent=2) + "\n"
)
PY

echo "=== TPU EVIDENCE RUN COMPLETED (status=$failed) ==="
exit "$failed"
