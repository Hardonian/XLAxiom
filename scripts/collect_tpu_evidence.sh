#!/usr/bin/env bash
# Automated Physical TPU Evidence Collector for XLAxiom
# Billed per second by Google Cloud; destroy the VM immediately after collection.
# Automatically tears down the TPU VM in a trap/exit block to prevent idle costs.

set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${2:-us-south1-a}"
ACCELERATOR_TYPE="${3:-v5litepod-4}"
VERSION="tpu-ubuntu2204-base"
VM_NAME="${4:-xlaxiom-tpu-eval}"
KEEP_VM="${5:-false}"

if [ -z "${PROJECT_ID}" ]; then
  echo "Error: No active GCP project. Run 'gcloud config set project <ID>' or pass PROJECT_ID as arg 1." >&2
  exit 1
fi

echo "=========================================================="
echo " XLAxiom Physical TPU Evidence Collector"
echo "=========================================================="
echo " Project:          ${PROJECT_ID}"
echo " Zone:             ${ZONE}"
echo " Accelerator:      ${ACCELERATOR_TYPE}"
echo " VM Name:          ${VM_NAME}"
echo " Estimated Cost:   depends on zone/pricing mode (destroyed on exit)"
echo "=========================================================="

# Ensure TPU API is enabled
gcloud services enable tpu.googleapis.com --project="${PROJECT_ID}"

VM_CREATED=0

cleanup() {
  if [ "${VM_CREATED}" -eq 1 ] && [ "${KEEP_VM}" != "true" ]; then
    echo "Tearing down Cloud TPU VM '${VM_NAME}' to prevent idle billing..."
    gcloud compute tpus tpu-vm delete "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --quiet || true
    echo "Cloud TPU VM successfully destroyed."
  fi
}
trap cleanup EXIT

echo "Creating Cloud TPU VM '${VM_NAME}'..."
gcloud compute tpus tpu-vm create "${VM_NAME}" \
    --zone="${ZONE}" \
    --accelerator-type="${ACCELERATOR_TYPE}" \
    --version="${VERSION}" \
    --project="${PROJECT_ID}" \
    --quiet
VM_CREATED=1

echo "Setting up JAX with libtpu and executing XLAxiom test suite on TPU..."
REMOTE_STATUS=0
gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command='
set -euo pipefail

sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

rm -rf XLAxiom
git clone https://github.com/Hardonian/XLAxiom.git
cd XLAxiom

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Verifying Physical TPU Silicon ==="
python -c "import jax; print(\"JAX Devices:\", jax.devices()); assert any(d.platform == \"tpu\" for d in jax.devices())"

rm -rf evidence/tpu
mkdir -p evidence/tpu
status_tsv=evidence/tpu/stage-status.tsv
: > "$status_tsv"
failed=0
run_stage() {
    stage="$1"
    shift
    set +e
    "$@" >"evidence/tpu/${stage}.log" 2>&1
    code=$?
    set -e
    printf "%s\t%s\n" "$stage" "$code" >> "$status_tsv"
    if [ "$code" -ne 0 ]; then failed=1; tail -n 20 "evidence/tpu/${stage}.log"; fi
}
run_stage doctor env PYTHONWARNINGS=ignore python -m xlaxiom.cli doctor
cp evidence/tpu/doctor.log evidence/tpu/doctor.json
run_stage environment python -m xlaxiom.cli environment capture --label "google-cloud-tpu-v5e-viperlite" --output evidence/tpu/environment.json
run_stage tpu-eager python -m xlaxiom.cli property run --x64 --backend jax-tpu-eager --output evidence/tpu/tpu-eager-evidence.json
run_stage tpu-jit python -m xlaxiom.cli property run --x64 --backend jax-tpu-jit --output evidence/tpu/tpu-jit-evidence.json
run_stage tpu-campaign python -m xlaxiom.cli property campaign --x64 --cases 50 --backend jax-tpu-jit --output evidence/tpu/tpu-campaign.json
run_stage cpu-jit env JAX_PLATFORMS=cpu python -m xlaxiom.cli property run --x64 --backend jax-jit --output evidence/tpu/cpu-jit-evidence.json
run_stage cpu-compare python -m xlaxiom.cli property compare \
    --baseline evidence/tpu/cpu-jit-evidence.json \
    --comparison evidence/tpu/tpu-jit-evidence.json \
    --output evidence/tpu/tpu-vs-cpu-comparison.json
run_stage higher-order python -m xlaxiom.cli gradient --higher-order --seed 0 --output evidence/tpu/tpu-higher-order.json
STATUS_TSV="$status_tsv" python -c "import json, os, pathlib; rows=[dict(stage=s, exit_code=int(c), passed=c == \"0\") for s,c in (line.split(chr(9),1) for line in pathlib.Path(os.environ[\"STATUS_TSV\"]).read_text().splitlines())]; pathlib.Path(\"evidence/tpu/stage-status.json\").write_text(json.dumps(dict(kind=\"physical-tpu-stage-status\", passed=all(r[\"passed\"] for r in rows), stages=rows), indent=2) + chr(10))"
exit "$failed"
' || REMOTE_STATUS=$?

echo "Downloading evidence artifacts to ./evidence/tpu/..."
mkdir -p evidence/tpu
gcloud compute tpus tpu-vm scp --recurse "${VM_NAME}:~/XLAxiom/evidence/tpu/*" ./evidence/tpu/ --zone="${ZONE}" --project="${PROJECT_ID}"

echo "Evidence artifacts collected in ./evidence/tpu/"
exit "${REMOTE_STATUS}"
