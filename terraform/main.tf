terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required GCP APIs
resource "google_project_service" "enabled_apis" {
  for_each = toset([
    "tpu.googleapis.com",
    "cloudkms.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "logging.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Cloud KMS KeyRing & Asymmetric SLSA Signer Key
resource "google_kms_key_ring" "xlaxiom_ring" {
  name     = "xlaxiom-keyring"
  location = var.location
  project  = var.project_id
  depends_on = [google_project_service.enabled_apis]
}

resource "google_kms_crypto_key" "slsa_signer" {
  name     = "slsa-signer"
  key_ring = google_kms_key_ring.xlaxiom_ring.id
  purpose  = "ASYMMETRIC_SIGN"

  version_template {
    algorithm        = "RSA_SIGN_PSS_3072_SHA256"
    protection_level = "HSM"
  }
}

# 3. Google Cloud Storage Evidence Bucket
resource "google_storage_bucket" "evidence_bucket" {
  name                        = "xlaxiom-evidence-${var.project_id}"
  location                    = var.dataset_location
  project                     = var.project_id
  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }
  depends_on = [google_project_service.enabled_apis]
}

# 4. BigQuery Telemetry Dataset & Tables
resource "google_bigquery_dataset" "xlaxiom_telemetry" {
  dataset_id  = "xlaxiom_telemetry"
  description = "XLAxiom continuous compiler and accelerator telemetry"
  location    = var.dataset_location
  project     = var.project_id
  depends_on  = [google_project_service.enabled_apis]
}

resource "google_bigquery_table" "campaigns" {
  dataset_id          = google_bigquery_dataset.xlaxiom_telemetry.dataset_id
  table_id            = "campaigns"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  schema = <<EOF
[
  {"name": "campaign_id", "type": "STRING", "mode": "REQUIRED"},
  {"name": "timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
  {"name": "environment", "type": "RECORD", "mode": "REQUIRED", "fields": [
    {"name": "platform", "type": "STRING", "mode": "REQUIRED"},
    {"name": "python_version", "type": "STRING", "mode": "REQUIRED"},
    {"name": "jax_version", "type": "STRING", "mode": "REQUIRED"},
    {"name": "accelerator", "type": "STRING", "mode": "REQUIRED"},
    {"name": "device_count", "type": "INTEGER", "mode": "REQUIRED"}
  ]},
  {"name": "total_cases", "type": "INTEGER", "mode": "REQUIRED"},
  {"name": "passed_cases", "type": "INTEGER", "mode": "REQUIRED"},
  {"name": "failed_cases", "type": "INTEGER", "mode": "REQUIRED"},
  {"name": "anomalies", "type": "INTEGER", "mode": "REQUIRED"},
  {"name": "duration_seconds", "type": "FLOAT", "mode": "REQUIRED"},
  {"name": "digest", "type": "STRING", "mode": "REQUIRED"}
]
EOF
}

# 5. Service Account for CI/CD and Headless Verification
resource "google_service_account" "xlaxiom_runner" {
  account_id   = "xlaxiom-runner"
  display_name = "XLAxiom Headless CI/CD Runner"
  project      = var.project_id
}

resource "google_project_iam_member" "kms_signer" {
  project = var.project_id
  role    = "roles/cloudkms.cryptoKeyVersionsSignerVerifier"
  member  = "serviceAccount:${google_service_account.xlaxiom_runner.email}"
}

resource "google_project_iam_member" "bigquery_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.xlaxiom_runner.email}"
}

resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.xlaxiom_runner.email}"
}

