#!/usr/bin/env bash
# ==============================================================================
# XLAxiom Multi-Generation Cloud TPU Fleet Evidence Collection Automation
# Supported Generations: TPU v4, TPU v5e, TPU v5p, TPU v6e (Trillium)
# Ensures zero-leak resource teardown with trap handlers.
# ==============================================================================

set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
if [ -z "${PROJECT_ID}" ]; then
  echo "Error: No GCP Project ID configured. Run 'gcloud config set project <ID>'." >&2
  exit 1
fi

echo "===================================================================="
echo " XLAxiom Multi-Generation Cloud TPU Fleet Evidence Collector"
echo " Target GCP Project: ${PROJECT_ID}"
echo "===================================================================="

# Fleet Matrix: (Generation, AcceleratorType, DefaultZone, OutputDir)
FLEET_CONFIGS=(
  "v4:v4-8:us-central2-b:evidence/tpu-v4"
  "v5e:v5litepod-4:us-south1-a:evidence/tpu-v5e"
  "v5p:v5p-8:us-east5-a:evidence/tpu-v5p"
  "v6e:v6e-4:us-central1-ba:evidence/tpu-v6e"
)

# Enable Cloud TPU API
gcloud services enable tpu.googleapis.com --project="${PROJECT_ID}" --quiet

for entry in "${FLEET_CONFIGS[@]}"; do
  IFS=":" read -r GEN ACCEL ZONE OUT_DIR <<< "${entry}"
  VM_NAME="xlaxiom-${GEN}-collector"
  
  echo ""
  echo ">>> Processing Generation: ${GEN} (${ACCEL}) in ${ZONE}..."
  mkdir -p "${OUT_DIR}"

  VM_CREATED=0
  cleanup() {
    if [ "${VM_CREATED}" -eq 1 ]; then
      echo "Tearing down Cloud TPU VM '${VM_NAME}' in ${ZONE}..."
      gcloud compute tpus tpu-vm delete "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --quiet || true
    fi
  }
  trap cleanup EXIT

  echo "Provisioning Cloud TPU VM '${VM_NAME}'..."
  if gcloud compute tpus tpu-vm create "${VM_NAME}" \
      --zone="${ZONE}" \
      --accelerator-type="${ACCEL}" \
      --version="tpu-ubuntu2204-base" \
      --project="${PROJECT_ID}" \
      --quiet; then
    VM_CREATED=1

    echo "Running XLAxiom qualification suite on remote TPU VM..."
    gcloud compute tpus tpu-vm ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command="
      set -euo pipefail
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      git clone https://github.com/Hardonian/XLAxiom.git ~/XLAxiom
      cd ~/XLAxiom
      uv sync --extra dev
      uv run python -m xlaxiom.cli cloud tpu-run --accelerator ${ACCEL} --cases 50 --output ~/results.json
    "

    echo "Downloading verification artifacts..."
    gcloud compute tpus tpu-vm scp "${VM_NAME}:~/results.json" "${OUT_DIR}/tpu-results.json" --zone="${ZONE}" --project="${PROJECT_ID}"
    
    # Clean up VM immediately
    cleanup
    VM_CREATED=0
    echo "Completed collection for ${GEN}."
  else
    echo "WARNING: Could not provision ${ACCEL} in ${ZONE} (quota or capacity unavailable). Skipping physical run."
  fi
  trap - EXIT
done

echo ""
echo "Multi-generation TPU evidence collection routine finished."

