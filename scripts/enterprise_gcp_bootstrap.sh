#!/usr/bin/env bash
# ==============================================================================
# XLAxiom Enterprise GCP Bootstrap Script (CLI-based)
# Provisions KMS Keyring, BigQuery Telemetry Dataset, GCS Bucket, and Views.
# ==============================================================================

set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
LOCATION="${2:-global}"
BQ_LOCATION="${3:-US}"

if [ -z "${PROJECT_ID}" ]; then
  echo "Error: GCP Project ID is required. Pass as arg 1 or set active project." >&2
  exit 1
fi

echo "===================================================================="
echo " XLAxiom Enterprise GCP Bootstrapper"
echo " Project: ${PROJECT_ID} | KMS Location: ${LOCATION} | BQ: ${BQ_LOCATION}"
echo "===================================================================="

# 1. Enable Services
echo ">>> Enabling required GCP services..."
gcloud services enable \
  tpu.googleapis.com \
  cloudkms.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  logging.googleapis.com \
  --project="${PROJECT_ID}" --quiet

# 2. Provision KMS Keyring and Asymmetric Key
echo ">>> Provisioning Cloud KMS Keyring and CryptoKey..."
if ! gcloud kms keyrings describe xlaxiom-keyring --location="${LOCATION}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud kms keyrings create xlaxiom-keyring --location="${LOCATION}" --project="${PROJECT_ID}"
fi

if ! gcloud kms keys describe slsa-signer --keyring=xlaxiom-keyring --location="${LOCATION}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud kms keys create slsa-signer \
    --keyring=xlaxiom-keyring \
    --location="${LOCATION}" \
    --purpose=asymmetric-signing \
    --default-algorithm=rsa-sign-pss-3072-sha256 \
    --protection-level=hsm \
    --project="${PROJECT_ID}"
fi

# 3. Create GCS Evidence Bucket
BUCKET_NAME="xlaxiom-evidence-${PROJECT_ID}"
echo ">>> Ensuring GCS bucket gs://${BUCKET_NAME} exists..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" --location="${BQ_LOCATION}" --project="${PROJECT_ID}" --uniform-bucket-level-access
fi

# 4. Create BigQuery Telemetry Dataset
echo ">>> Provisioning BigQuery telemetry dataset..."
bq --location="${BQ_LOCATION}" mk --dataset --project_id="${PROJECT_ID}" --description "XLAxiom Telemetry" xlaxiom_telemetry || true

# 5. Deploy Looker Studio & Analytics Views
echo ">>> Deploying Looker Studio analytical SQL views..."
sed "s/{PROJECT_ID}/${PROJECT_ID}/g" sql/bigquery_views.sql > /tmp/resolved_views.sql
bq query --use_legacy_sql=false --project_id="${PROJECT_ID}" < /tmp/resolved_views.sql || true
rm -f /tmp/resolved_views.sql

echo ""
echo "XLAxiom Enterprise GCP infrastructure successfully bootstrapped!"

